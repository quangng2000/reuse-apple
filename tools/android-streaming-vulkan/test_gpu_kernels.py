#!/usr/bin/env python3
"""Exercise actual FP32 Vulkan shader arithmetic through the native GPU engine.

The generated miniature plans use the same 33-port/31-state transport as the
real model. Outputs are read from state slots so tests can inspect arbitrary
tensor sizes, not just the model's 321-bin spectra. No ORT executes on the GPU.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent
F32 = np.float32

def dims6(shape):
    return [1]*(6-len(shape))+list(shape)

def strides6(shape, output):
    shape=dims6(shape);output=dims6(output)
    strides=[0]*6;stride=1
    for axis in range(5,-1,-1):
        strides[axis]=0 if shape[axis]==1 and output[axis]!=1 else stride
        stride*=shape[axis]
    return strides

class Case:
    def __init__(self,name,opcode,expected,extra=None):
        self.name=name;self.arrays=[];self.cursor=0
        self.expected=np.asarray(expected,dtype=np.float32).reshape(-1)
        self.extra=None if extra is None else np.asarray(extra,dtype=np.float32).reshape(-1)
        self.words=[0]*32;self.words[0]=opcode;self.words[2]=self.expected.size
        self.address_mask=1<<1
        self.output_shift=0;self.extra_field=11;self.require_bitwise=False;self.state_input_overrides={}
    def alloc(self,values):
        values=np.asarray(values,dtype=np.float32).reshape(-1)
        self.cursor=(self.cursor+63)//64*64
        offset=self.cursor;self.arrays.append((offset,values));self.cursor+=len(values)
        return offset
    def input(self,field,values):
        offset=self.alloc(values);self.words[field]=offset;self.address_mask|=1<<field
        return offset
    def write(self,directory):
        directory.mkdir(parents=True,exist_ok=True)
        ports_in=[(self.alloc(np.zeros(321)),321),(self.alloc(np.zeros(321)),321)]
        ports_out=[(self.alloc(np.zeros(321)),321),(self.alloc(np.zeros(321)),321)]
        states=[]
        for i in range(31):
            count=len(self.expected) if i==0 else len(self.extra) if i==1 and self.extra is not None else 1
            initial=np.zeros(count) if i not in self.state_input_overrides else self.state_input_overrides[i][1]
            assert len(np.asarray(initial).reshape(-1))==count
            source=self.alloc(initial);target=self.alloc(np.zeros(count))
            if i in self.state_input_overrides:self.words[self.state_input_overrides[i][0]]=source
            ports_in.append((source,count));ports_out.append((target,count));states.append((source,target,count))
            if i==0:self.words[1]=target+self.output_shift
            if i==1 and self.extra is not None:self.words[self.extra_field]=target;self.address_mask|=1<<self.extra_field
        count=(self.cursor+63)//64*64
        values=np.zeros(count,dtype='<f4')
        for offset,array in self.arrays:values[offset:offset+len(array)]=array
        payload=b'RVKPLAN1'+struct.pack('<6I',1,count,1,33,33,31)
        for item in ports_in+ports_out:payload+=struct.pack('<2I',*item)
        for item in states:payload+=struct.pack('<3I',*item)
        payload+=struct.pack('<33I',*[int(x)&0xffffffff for x in self.words],self.address_mask)
        (directory/'plan.bin').write_bytes(payload);values.tofile(directory/'weights.bin')
        np.zeros(642,dtype='<f4').tofile(directory/'input.f32')

def cases():
    rng=np.random.default_rng(1068)
    x=rng.uniform(-9,9,257).astype(F32)
    for kind,name,expected in [(1,'neg',-x),(2,'exp',np.exp(x)),(3,'sigmoid',1/(1+np.exp(-x))),
                               (4,'softplus',np.logaddexp(x,0)),(5,'atan',np.arctan(x)),
                               (14,'silu',x/(1+np.exp(-x)))]:
        c=Case(name,1,expected);c.input(3,x);c.words[6]=kind;yield c
    a=rng.normal(size=(2,3,1,5)).astype(F32);b=rng.normal(size=(1,3,4,1)).astype(F32)
    shape=(2,3,4,5)
    for kind,name,expected in [(1,'add-broadcast',a+b),(2,'mul-broadcast',a*b),(3,'div-broadcast',a/b),
                               (4,'prelu-broadcast',np.where(a<0,a*b,a+np.zeros_like(b))),
                               (5,'less-broadcast',a<b),(6,'greater-broadcast',a>b),(7,'equal-broadcast',a==b)]:
        c=Case(name,2,expected);c.input(3,a);c.input(4,b);c.words[6]=kind
        c.words[8:14]=dims6(shape);c.words[14:20]=strides6(a.shape,shape);c.words[20:26]=strides6(b.shape,shape);yield c
    condition=(a>0).astype(F32);out=np.where(condition!=0,a,b)
    c=Case('where-broadcast',3,out);c.input(3,condition);c.input(4,a);c.input(5,b)
    c.words[8:14]=dims6(shape);c.words[14:20]=strides6(condition.shape,shape)
    c.words[20:26]=strides6(a.shape,shape);c.words[26:32]=strides6(b.shape,shape);yield c
    a=np.arange(2*3*4*5,dtype=F32).reshape(2,3,4,5);out=a.transpose(2,0,3,1)
    c=Case('transpose-rank4',4,out);c.input(3,a);c.words[8:14]=dims6(out.shape)
    c.words[14:20]=[0,0,5,60,1,20];yield c
    out=a[:,::-1,1::2,::-2]
    c=Case('slice-negative',4,out);base=c.input(3,a);c.words[3]=base+2*20+5+4
    c.words[8:14]=dims6(out.shape);c.words[14:20]=[0,0,60,-20,10,-2];yield c
    a=np.arange(2*3*4*5,dtype=F32).reshape(2,3,4,5);pads=((0,0),(1,2),(2,0),(1,1))
    out=np.pad(a,pads,constant_values=-3.5);c=Case('pad-rank4',5,out);c.input(3,a)
    c.words[6]=struct.unpack('<I',struct.pack('<f',-3.5))[0]
    c.words[8:14]=dims6(out.shape);c.words[14:20]=dims6(a.shape);c.words[20:26]=[0,0]+[p[0] for p in pads];yield c
    a=np.arange(24,dtype=F32).reshape(2,1,3,1,4,1);permutation=(3,0,5,2,1,4);out=a.transpose(permutation)
    c=Case('transpose-rank6',4,out);c.input(3,a);c.words[8:14]=list(out.shape)
    raw_strides=[s//4 for s in a.strides];c.words[14:20]=[raw_strides[p] for p in permutation];yield c
    a=np.arange(6,dtype=F32).reshape(2,3);out=np.zeros((2,5),dtype=F32);out[:,1:4]=a
    c=Case('scatter-concat-part',6,out);c.input(3,a);c.output_shift=1;c.words[2]=a.size
    c.words[8:14]=dims6(a.shape);c.words[14:20]=[0,0,0,0,5,1];yield c
    # Fused concat/cache copy must preserve payload bits, including signed zero.
    for outer,first,second,drop,cache in [(1,5,3,0,False),(3,7,5,5,True),(2,1,3,3,True),(5,17,11,11,True)]:
        a=rng.normal(size=(outer,first)).astype(F32);b=rng.normal(size=(outer,second)).astype(F32)
        a[0,0]=F32(-0.0);b[-1,-1]=F32(-0.0)
        out=np.concatenate((a,b),axis=1);tail=out[:,drop:].copy() if cache else None
        c=Case(f'concat-cache-{outer}-{first}-{second}-{drop}-{cache}',13,out,tail)
        c.input(3,a);c.input(4,b);c.extra_field=5;c.require_bitwise=True
        if cache:c.state_input_overrides[1]=(3,a)
        c.words[6:11]=[first,second,outer,drop,int(cache)];yield c
    # Exact shape family used by temporal dense3x3; includes odd channel/tile tails.
    for n,cin,cout,width,dilation in [(1,2,17,1,1),(2,7,19,33,2),(1,64,64,161,4),(1,128,64,33,8)]:
        hin=1+2*dilation;a=rng.normal(size=(n,cin,hin,width)).astype(F32)
        w=rng.normal(size=(cout,cin,3,3)).astype(F32)*F32(.1);bias=rng.normal(size=cout).astype(F32)
        out=np.zeros((n,cout,width),dtype=F32)
        padded=np.pad(a,((0,0),(0,0),(0,0),(1,1)))
        for ci in range(cin):
            for ky in range(3):
                for kx in range(3):
                    # Double intermediates emulate one explicit float32 FMA.
                    product=w[None,:,ci,ky,kx,None].astype(np.float64)*padded[:,ci,ky*dilation,kx:kx+width][:,None,:].astype(np.float64)
                    out=(product+out.astype(np.float64)).astype(F32)
        out=out+bias[None,:,None]
        c=Case(f'conv-dense3x3-{n}-{cin}-{cout}-{width}-{dilation}',8,out)
        c.input(3,a);c.input(4,w);c.input(5,bias)
        c.words[6:25]=[n,cout,1,width,n,cin,hin,width,3,3,1,1,0,1,dilation,1,1,1,0];yield c
    for m,n,k in [(13,19,23),(161,128,64),(7,5,3)]:
        a=rng.normal(size=(m,k)).astype(F32);b=rng.normal(size=(k,n)).astype(F32)
        # Reference follows the shader's serialized product + add boundaries.
        out=np.zeros((m,n),dtype=F32)
        for j in range(k):out=out+a[:,j:j+1]*b[j:j+1,:]
        c=Case(f'matmul-{m}-{n}-{k}',7,out);c.input(3,a);c.input(4,b);c.words[6:14]=[m,n,k,1,0,0,0,0];yield c
    m,n,k=7,11,9;a=rng.normal(size=(k,m)).astype(F32);b=rng.normal(size=(n,k)).astype(F32)
    out=np.zeros((m,n),dtype=F32)
    for j in range(k):out=out+a[j,:,None]*b[:,j][None,:]
    c=Case('matmul-transposed',7,out);c.input(3,a);c.input(4,b);c.words[6:14]=[m,n,k,1,0,0,1,1];yield c
    for mode in [1,2]:
        m,n,k=13,19,23;a=rng.normal(size=(m,k)).astype(F32);b=rng.normal(size=(k,n)).astype(F32)
        bias=rng.normal(size=n if mode==1 else (m,n)).astype(F32)
        out=np.zeros((m,n),dtype=F32)
        for j in range(k):out=out+a[:,j:j+1]*b[j:j+1,:]
        out=out+bias
        c=Case(f'matmul-fused-bias-{mode}',7,out);c.input(3,a);c.input(4,b);c.input(5,bias)
        c.words[6:15]=[m,n,k,1,0,0,0,0,mode];yield c
        c=Case(f'matmul-fused-bias-softplus-{mode}',7,np.logaddexp(out,F32(0)))
        c.input(3,a);c.input(4,b);c.input(5,bias);c.words[6:16]=[m,n,k,1,0,0,0,0,mode,1];yield c
    a=rng.normal(size=(3,17)).astype(F32);scale=rng.normal(size=17).astype(F32);bias=rng.normal(size=17).astype(F32)
    mean=np.mean(a,axis=-1,keepdims=True);var=np.mean((a-mean)**2,axis=-1,keepdims=True)
    out=(a-mean)/np.sqrt(var+F32(1e-5))*scale+bias
    c=Case('layernorm',9,out);c.input(3,a);c.input(4,scale);c.input(5,bias);c.words[6:11]=[3,17,struct.unpack('<I',struct.pack('<f',1e-5))[0],1,1];yield c
    a=rng.normal(size=(2,17,3,4)).astype(F32);scale=rng.normal(size=17).astype(F32)
    bias=rng.normal(size=17).astype(F32);slope=rng.uniform(.01,.5,size=17).astype(F32)
    mean=np.mean(a,axis=1,keepdims=True);variance=np.mean((a-mean)**2,axis=1,keepdims=True)
    out=(a-mean)/np.sqrt(variance+F32(1e-5))*scale[None,:,None,None]+bias[None,:,None,None]
    out=np.where(out<0,out*slope[None,:,None,None],out)
    c=Case('layernorm-nchw-prelu',9,out);c.input(3,a);c.input(4,scale);c.input(5,bias);c.input(14,slope)
    c.words[6:14]=[24,17,struct.unpack('<I',struct.pack('<f',1e-5))[0],1,1,1,12,1];c.words[15]=17;yield c
    for group in [1,2,4]:
        n,cin,hin,win,cout,kh,kw=2,4,5,7,8,3,3;hout,wout=3,7
        a=rng.normal(size=(n,cin,hin,win)).astype(F32)
        w=rng.normal(size=(cout,cin//group,kh,kw)).astype(F32);bias=rng.normal(size=cout).astype(F32)
        out=np.zeros((n,cout,hout,wout),dtype=F32)
        for batch in range(n):
            for oc in range(cout):
                for oy in range(hout):
                    for ox in range(wout):
                        total=F32(0)
                        for ci in range(cin//group):
                            ic=(oc//(cout//group))*(cin//group)+ci
                            for ky in range(kh):
                                for kx in range(kw):
                                    iy=oy*2+ky-1;ix=ox+kx-1
                                    if 0<=iy<hin and 0<=ix<win:total=F32(total+F32(a[batch,ic,iy,ix]*w[oc,ci,ky,kx]))
                        out[batch,oc,oy,ox]=F32(total+bias[oc])
        c=Case(f'conv-group-{group}',8,out);c.input(3,a);c.input(4,w);c.input(5,bias)
        c.words[6:25]=[n,cout,hout,wout,n,cin,hin,win,kh,kw,2,1,1,1,1,1,group,1,0];yield c
        nhwc=out.transpose(0,2,3,1).copy();expected=nhwc/(1+np.exp(-nhwc))
        c=Case(f'conv-nhwc-silu-group-{group}',8,expected)
        c.input(3,a.transpose(0,2,3,1).copy());c.input(4,w);c.input(5,bias)
        c.words[6:27]=[n,cout,hout,wout,n,cin,hin,win,kh,kw,2,1,1,1,1,1,group,1,0,1,1];yield c
    n,cin,hin,win,cout,kh,kw=1,3,3,4,4,3,3;hout,wout=5,7
    a=rng.normal(size=(n,cin,hin,win)).astype(F32);w=rng.normal(size=(cin,cout,kh,kw)).astype(F32)
    out=np.zeros((n,cout,hout,wout),dtype=F32)
    for oc in range(cout):
        for oy in range(hout):
            for ox in range(wout):
                total=F32(0)
                for ic in range(cin):
                    for ky in range(kh):
                        for kx in range(kw):
                            ny=oy+1-ky;nx=ox+1-kx
                            if ny%2 or nx%2:continue
                            iy=ny//2;ix=nx//2
                            if 0<=iy<hin and 0<=ix<win:total=F32(total+F32(a[0,ic,iy,ix]*w[ic,oc,ky,kx]))
                out[0,oc,oy,ox]=total
    c=Case('conv-transpose',8,out);c.input(3,a);c.input(4,w)
    c.words[6:25]=[n,cout,hout,wout,n,cin,hin,win,kh,kw,2,2,1,1,1,1,1,0,1];yield c
    bins,channels,components=5,32,16
    for temporal in [False,True]:
        old=rng.normal(size=(bins if temporal else 1,channels,components)).astype(F32)*F32(.1)
        u=rng.normal(size=(bins,channels)).astype(F32);delta=rng.uniform(.001,.1,size=(bins,channels)).astype(F32)
        b=rng.normal(size=(bins,components)).astype(F32);cc=rng.normal(size=(bins,components)).astype(F32)
        gate=rng.uniform(.01,.8,size=(bins,channels)).astype(F32);coef=-rng.uniform(.1,2,size=(channels,components)).astype(F32)
        skip=rng.normal(size=channels).astype(F32);out=np.empty((bins,channels),dtype=F32)
        next_state=old.copy()
        for t in range(bins):
            prior=old[t] if temporal else next_state[0]
            updated=np.exp(delta[t,:,None]*coef)*prior+(delta[t,:,None]*b[t])*u[t,:,None]
            next_state[t if temporal else 0]=updated
            # Same binary-tree FP32 reduction as the subgroup shader.
            values=(updated*cc[t]).copy()
            for stride in [8,4,2,1]:values[:,:stride]=values[:,:stride]+values[:,stride:stride*2]
            out[t]=(values[:,0]+skip*u[t])*gate[t]
        c=Case('temporal' if temporal else 'frequency',11 if temporal else 10,out,next_state)
        for field,value in [(3,old),(4,u),(5,delta),(6,b),(7,cc),(8,gate),(9,coef),(10,skip)]:c.input(field,value)
        c.words[12:15]=[bins,channels,components];yield c

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli',type=Path,default=ROOT/'generated/build-host/reuse-vulkan-cli')
    parser.add_argument('--shaders',type=Path,default=ROOT/'generated/shaders')
    parser.add_argument('--output',type=Path,default=ROOT/'generated/kernel-tests')
    parser.add_argument('--tiled-matmul',action='store_true')
    parser.add_argument('--tiled-conv',action='store_true')
    parser.add_argument('--expect-dense-conv-pipeline',action='store_true',help='Require dedicated conv_3x3_tiled selection in shaped convolution cases.')
    parser.add_argument('--case-prefix',default='',help='Run only cases with this name prefix (finite-output rejection still runs).')
    args=parser.parse_args();results=[]
    for case in cases():
        if not case.name.startswith(args.case_prefix):continue
        if args.expect_dense_conv_pipeline and case.name.startswith('conv-dense3x3'):case.words[27]=1
        directory=args.output/case.name;case.write(directory)
        command=[str(args.cli),str(directory/'plan.bin'),str(directory/'weights.bin'),str(args.shaders),
                 str(directory/'input.f32'),str(directory/'output.f32'),'--states',str(directory/'states')]
        if args.tiled_matmul:command.append('--tiled-matmul')
        if args.tiled_conv:command.append('--tiled-conv')
        if args.expect_dense_conv_pipeline and case.name.startswith('conv-dense3x3'):command.append('--profile')
        result=subprocess.run(command,capture_output=True,text=True,timeout=90)
        (directory/'stdout.txt').write_text(result.stdout);(directory/'stderr.txt').write_text(result.stderr)
        if result.returncode:raise RuntimeError(f'{case.name}: {result.stderr[-4000:]}')
        receipt=json.loads(result.stdout)
        if args.expect_dense_conv_pipeline and case.name.startswith('conv-dense3x3'):
            assert receipt['profile']['operations'][0]['kernel']=='conv_3x3_tiled','Dedicated dense pipeline was not selected'
        actual=np.fromfile(directory/'states/state-0.f32',dtype='<f4')
        np.testing.assert_allclose(actual,case.expected,rtol=3e-6,atol=3e-6,err_msg=case.name)
        if case.require_bitwise:np.testing.assert_array_equal(actual.view(np.uint32),case.expected.view(np.uint32),err_msg=case.name)
        extra_error=0.
        if case.extra is not None:
            extra=np.fromfile(directory/'states/state-1.f32',dtype='<f4')
            np.testing.assert_allclose(extra,case.extra,rtol=3e-6,atol=3e-6,err_msg=case.name+' state')
            extra_error=float(np.max(np.abs(extra-case.extra)))
            if case.require_bitwise:np.testing.assert_array_equal(extra.view(np.uint32),case.extra.view(np.uint32),err_msg=case.name+' cache')
        item={'name':case.name,'values':actual.size,'maxAbs':float(np.max(np.abs(actual-case.expected))),
              'stateMaxAbs':extra_error,'device':receipt['device']}
        if receipt.get('profile'):item['observedKernel']=receipt['profile']['operations'][0]['kernel']
        print(json.dumps({k:v for k,v in item.items() if k!='device'}),flush=True);results.append(item)
    # Deliberate GPU overflow must be caught by the GPU-wide finite check,
    # without exposing/committing a failed recurrent output to the caller.
    case=Case('reject-nonfinite-gpu-output',1,np.zeros(1,dtype=F32));case.input(3,[100.]);case.words[6]=2
    directory=args.output/case.name;case.write(directory)
    command=[str(args.cli),str(directory/'plan.bin'),str(directory/'weights.bin'),str(args.shaders),
             str(directory/'input.f32'),str(directory/'output.f32')]
    result=subprocess.run(command,capture_output=True,text=True,timeout=90)
    (directory/'stderr.txt').write_text(result.stderr)
    assert result.returncode!=0 and 'nonfinite output or recurrent state' in result.stderr,result.stderr
    assert not (directory/'output.f32').exists(),'Failed GPU result was exposed as successful output'
    print(json.dumps({'negativeCase':case.name,'rejected':True}),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'passed':len(results),'negativeGpuOutputRejected':True,'tests':results},indent=2)+'\n')
if __name__=='__main__':main()
