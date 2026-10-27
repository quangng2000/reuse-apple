"""Independent CPU execution of the fixed plan for layout/semantic validation.

This is a development oracle, not an Android runtime fallback. Torch is used
only for FP32 convolution and NumPy for all other operators. The Vulkan engine
loads the same plan without any Torch/ONNX/ORT dependency.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np
import torch
import torch.nn.functional as F

from export_plan import HERE, SOURCE, digest


class Plan:
    def __init__(self, directory):
        directory=Path(directory)
        b=(directory/'plan.bin').read_bytes()
        assert b[:8]==b'RVKPLAN1'
        version,n,ops,ni,no,ns=struct.unpack_from('<6I',b,8)
        assert version==1
        cursor=32
        def records(count,width):
            nonlocal cursor
            out=np.frombuffer(b,dtype='<u4',count=count*width,offset=cursor).reshape(count,width)
            cursor+=count*width*4
            return out
        self.inputs=records(ni,2);self.outputs=records(no,2);self.states=records(ns,3);self.ops=records(ops,33)
        assert cursor==len(b)
        self.initial=np.fromfile(directory/'weights.bin',dtype='<f4')
        assert len(self.initial)==n
        self.arena=self.initial.copy()
        self.audit=json.loads((directory/'export.json').read_text())
        self.shapes={x['name']:tuple(x['shape']) for x in self.audit['inputs']+self.audit['outputs']}

    def reset(self): self.arena[:]=self.initial

    def view(self, offset,shape,strides=None):
        return np.ndarray(shape=tuple(int(v) for v in shape),dtype='<f4',buffer=self.arena,offset=int(offset)*4,
                          strides=None if strides is None else tuple(int(v)*4 for v in strides))

    def execute(self,p):
        a=self.arena
        out=int(p[1]);count=int(p[2]);x=int(p[3]);y=int(p[4]);z=int(p[5]);op=int(p[0]);kind=int(p[6])
        ov=a[out:out+count]
        if op==1:
            xv=a[x:x+count]
            if kind==1:np.negative(xv,out=ov)
            elif kind==2:np.exp(xv,out=ov)
            elif kind==3:ov[:]=1/(1+np.exp(-xv))
            elif kind==4:ov[:]=np.maximum(xv,0)+np.log1p(np.exp(-np.abs(xv)))
            elif kind==5:np.arctan(xv,out=ov)
            elif kind==14:ov[:]=xv*(1/(1+np.exp(-xv)))
            else:raise ValueError(kind)
        elif op in (2,3):
            shape=p[8:14]
            xx=self.view(x,shape,p[14:20]);yy=self.view(y,shape,p[20:26]);oo=ov.reshape(shape)
            if op==3:oo[:]=np.where(xx!=0,yy,self.view(z,shape,p[26:32]))
            elif kind==1:np.add(xx,yy,out=oo)
            elif kind==2:np.multiply(xx,yy,out=oo)
            elif kind==3:np.divide(xx,yy,out=oo)
            elif kind==4:oo[:]=np.where(xx<0,xx*yy,xx)
            elif kind==5:oo[:]=xx<yy
            elif kind==6:oo[:]=xx>yy
            elif kind==7:oo[:]=xx==yy
            else:raise ValueError(kind)
        elif op==4:
            s=np.array(p[14:20],dtype=np.uint32).view(np.int32)
            ov[:]=self.view(x,p[8:14],s).reshape(-1)
        elif op==5:
            shape=p[8:14];ishape=p[14:20];beg=p[20:26]
            padding=tuple((int(b),int(o-i-b)) for o,i,b in zip(shape,ishape,beg))
            ov[:]=np.pad(self.view(x,ishape),padding).reshape(-1)
        elif op==6:
            self.view(out,p[8:14],p[14:20])[:]=a[x:x+count].reshape(p[8:14])
        elif op==7:
            m,n,k,batch,ast,bst,ta,tb=[int(v) for v in p[6:14]]
            for i in range(batch):
                aa=a[x+i*ast:x+i*ast+m*k].reshape((k,m) if ta else (m,k));bb=a[y+i*bst:y+i*bst+k*n].reshape((n,k) if tb else (k,n))
                if ta:aa=aa.T
                if tb:bb=bb.T
                result=np.matmul(aa,bb)
                if p[14]==1:result+=a[z:z+n]
                elif p[14]==2:result+=a[z+i*m*n:z+(i+1)*m*n].reshape(m,n)
                if p[15]:result=np.maximum(result,0)+np.log1p(np.exp(-np.abs(result)))
                ov[i*m*n:(i+1)*m*n]=result.reshape(-1)
        elif op==8:
            bn,co,ho,wo,ni,ci,hi,wi,kh,kw,sh,sw,ph,pw,dh,dw,group,hb,tr=[int(v) for v in p[6:25]]
            assert tr==0 and bn==ni
            xx=a[x:x+ni*ci*hi*wi]
            xx=xx.reshape(ni,hi,wi,ci).transpose(0,3,1,2) if p[25] else xx.reshape(ni,ci,hi,wi)
            xx=torch.from_numpy(xx);ww=torch.from_numpy(a[y:y+co*(ci//group)*kh*kw].reshape(co,ci//group,kh,kw))
            bias=torch.from_numpy(a[z:z+co]) if hb else None
            result=F.conv2d(xx,ww,bias,stride=(sh,sw),padding=(ph,pw),dilation=(dh,dw),groups=group)
            if p[25]:result=result[:,:,:ho,:wo]
            assert tuple(result.shape)==(bn,co,ho,wo),(result.shape,(bn,co,ho,wo))
            result=result.numpy()
            if p[25]:result=result.transpose(0,2,3,1)
            if p[26]:result=result*(1/(1+np.exp(-result)))
            ov[:]=result.reshape(-1)
        elif op==9:
            rows,cols=int(p[6]),int(p[7]);eps=struct.unpack('<f',struct.pack('<I',int(p[8])))[0]
            if p[11]:
                spatial=int(p[12]);batch=rows//spatial
                xx=a[x:x+rows*cols].reshape(batch,cols,spatial).transpose(0,2,1).reshape(rows,cols)
            else:xx=a[x:x+rows*cols].reshape(rows,cols)
            mean=np.mean(xx,axis=1,keepdims=True,dtype=np.float32)
            variance=np.mean((xx-mean)**2,axis=1,keepdims=True,dtype=np.float32)
            yy=(xx-mean)/np.sqrt(variance+np.float32(eps))
            if p[9]:yy*=a[y:y+cols]
            if p[10]:yy+=a[z:z+cols]
            if p[13]:
                slope=a[int(p[14]):int(p[14])+int(p[15])]
                yy=np.where(yy<0,yy*slope,yy)
            if p[11]:yy=yy.reshape(batch,spatial,cols).transpose(0,2,1)
            ov[:]=yy.reshape(-1)
        elif op in (10,11):
            old,u,delta,b,c,gate,coeff,skip,new,bins,ch,ss=[int(v) for v in p[3:15]]
            av=a[coeff:coeff+ch*ss].reshape(ch,ss);dv=a[skip:skip+ch]
            uv=a[u:u+bins*ch].reshape(bins,ch);dt=a[delta:delta+bins*ch].reshape(bins,ch)
            bv=a[b:b+bins*ss].reshape(bins,ss);cv=a[c:c+bins*ss].reshape(bins,ss);gv=a[gate:gate+bins*ch].reshape(bins,ch)
            if op==10:
                state=a[old:old+ch*ss].reshape(ch,ss).copy();oo=ov.reshape(bins,ch)
                for i in range(bins):
                    state=np.exp(dt[i,:,None]*av)*state+(dt[i,:,None]*bv[i])*uv[i,:,None]
                    summed=np.zeros(ch,dtype=np.float32)
                    for j in range(ss):summed+=state[:,j]*cv[i,j]
                    oo[i]=(summed+dv*uv[i])*gv[i]
                a[new:new+ch*ss]=state.reshape(-1)
            else:
                prev=a[old:old+bins*ch*ss].reshape(bins,ch,ss)
                state=np.exp(dt[:,:,None]*av)*prev+(dt[:,:,None]*bv[:,None,:])*uv[:,:,None]
                summed=np.zeros((bins,ch),dtype=np.float32)
                for j in range(ss):summed+=state[:,:,j]*cv[:,j,None]
                ov[:]=((summed+dv*uv)*gv).reshape(-1)
                a[new:new+bins*ch*ss]=state.reshape(-1)
        elif op==13:
            first,second,outer,drop,has_cache=[int(v) for v in p[6:11]]
            result=np.concatenate([a[x:x+first*outer].reshape(outer,first),
                                   a[y:y+second*outer].reshape(outer,second)],axis=1)
            ov[:]=result.reshape(-1)
            if has_cache:a[z:z+(first+second-drop)*outer]=result[:,drop:].reshape(-1)
        else:raise ValueError(op)

    def run(self, magnitude,phase, *, trace=None):
        for data,(offset,count) in zip((magnitude,phase),self.inputs[:2]):
            self.arena[offset:offset+count]=data.reshape(-1)
        with np.errstate(all='ignore'):
            for i,p in enumerate(self.ops):
                self.execute(p)
                if trace is not None:trace(i,p,self)
        result=[self.arena[o:o+n].copy() for o,n in self.outputs]
        for old,new,n in self.states:self.arena[old:old+n]=self.arena[new:new+n]
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,default=HERE/'generated');parser.add_argument('--frames',type=int,default=3)
    args=parser.parse_args();torch.set_num_threads(1)
    import onnxruntime as ort
    assert ort.__version__=='1.23.2'
    options=ort.SessionOptions();options.intra_op_num_threads=1
    lib=SOURCE.parent/'build-host/libreuse_streaming_stable_v2.dylib'
    options.register_custom_ops_library(str(lib))
    session=ort.InferenceSession(str(SOURCE),options,providers=['CPUExecutionProvider'])
    plan=Plan(args.plan);rng=np.random.default_rng(728)
    states={v.name:np.zeros(v.shape,dtype=np.float32) for v in session.get_inputs()[2:]}
    stats=[]
    for frame in range(args.frames):
        mag=np.abs(rng.normal(0,.05,(1,321,1))).astype(np.float32);phase=rng.uniform(-3,3,(1,321,1)).astype(np.float32)
        expected=session.run(None,{'magnitude':mag,'phase':phase,**states})
        t=time.monotonic();actual=plan.run(mag,phase);elapsed=time.monotonic()-t
        errors=[]
        for info,ref,got in zip(session.get_outputs(),expected,actual):
            err=np.asarray(got,dtype=np.float64)-ref.reshape(-1)
            errors.append({'name':info.name,'maxAbs':float(np.max(np.abs(err))),'relativeRms':float(np.linalg.norm(err)/max(np.linalg.norm(ref.reshape(-1)),1e-30))})
        for i,info in enumerate(session.get_outputs()[2:]):
            states['cache_'+info.name[5:]]=expected[i+2]
        stats.append({'frame':frame,'seconds':elapsed,'outputs':errors})
        print(json.dumps({'frame':frame,'seconds':elapsed,'maxError':max(x['maxAbs'] for x in errors),'spectra':errors[:2]}),flush=True)
    (args.plan/'cpu-plan-validation.json').write_text(json.dumps({'frames':args.frames,'sourceSha256':digest(SOURCE),'framesDetail':stats},indent=2)+'\n')


if __name__=='__main__':main()
