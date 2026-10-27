"""Focused compilation, storage lifetime, ABI and semantic regressions."""
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import onnx

import export_plan as ex


class ExportPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=onnx.load(ex.SOURCE)
        cls.tensors,cls.nodes=ex.infer(cls.model)
        cls.nodes=ex.optimize(ex.live_nodes(cls.model,cls.tensors,cls.nodes),cls.tensors)
        cls.arena,cls.memory=ex.allocate(cls.model,cls.tensors,cls.nodes)
        cls.ops,cls.audit=ex.descriptors(cls.tensors,cls.nodes)

    def test_source_is_exact_pinned_fp32_model(self):
        self.assertEqual(ex.digest(ex.SOURCE),ex.SOURCE_SHA)
        self.assertEqual(len(self.model.graph.input),33)
        self.assertEqual(len(self.model.graph.output),33)
        float_weights=[i for i in self.model.graph.initializer if not i.name.startswith('constant')]
        self.assertTrue(all(i.data_type in (onnx.TensorProto.FLOAT,onnx.TensorProto.INT64) for i in float_weights))
        self.assertEqual(sum(n.op_type=='TemporalStepV2' for n,a in self.nodes),8)
        self.assertEqual(sum(n.op_type=='FrequencyScanV2' for n,a in self.nodes),16)

    def test_all_graph_outputs_match_declared_shape(self):
        for v in self.model.graph.output:
            declared=tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
            self.assertEqual(self.tensors[v.name].shape,declared)

    def test_all_persistent_inputs_outputs_disjoint(self):
        roots={ex.root_name(self.tensors,v.name) for v in list(self.model.graph.input)+list(self.model.graph.output)}
        spans=sorted((self.tensors[n].offset,self.tensors[n].offset+self.tensors[n].count,n) for n in roots)
        for a,b in zip(spans,spans[1:]): self.assertLessEqual(a[1],b[0],(a,b))

    def test_live_tensors_never_overlap(self):
        # Each operation writes distinct storage while all of its inputs are
        # still live; aliases count once, graph outputs survive the last op.
        use={}
        for i,(n,a) in enumerate(self.nodes):
            for name in n.input:
                if name:use[ex.root_name(self.tensors,name)]=i
        for v in self.model.graph.output:use[ex.root_name(self.tensors,v.name)]=len(self.nodes)
        live={ex.root_name(self.tensors,v.name) for v in self.model.graph.input}
        live|={ex.root_name(self.tensors,name) for n,a in self.nodes for name in n.input if name and self.tensors[ex.root_name(self.tensors,name)].constant is not None}
        for i,(n,a) in enumerate(self.nodes):
            live.update(n.output)
            spans=sorted((self.tensors[name].offset,self.tensors[name].offset+self.tensors[name].count,name) for name in live)
            for left,right in zip(spans,spans[1:]):self.assertLessEqual(left[1],right[0],(i,left,right))
            live={name for name in live if use.get(name,-1)>i or self.tensors[name].constant is not None}

    def test_arena_is_bounded_and_constants_are_finite(self):
        self.assertLessEqual(self.arena.nbytes,128*1024**2)
        self.assertTrue(np.isfinite(self.arena).all())
        self.assertGreater(self.memory['scratchFloats'],0)

    def test_fp32_constants_copy_bitwise_without_quantization(self):
        for tensor in self.tensors.values():
            if tensor.constant is None or tensor.offset<0 or tensor.constant.dtype!=np.float32:continue
            actual=self.arena[tensor.offset:tensor.offset+tensor.count]
            self.assertTrue(np.array_equal(actual.view(np.uint32),tensor.constant.reshape(-1).view(np.uint32)),tensor.name)

    def test_every_masked_address_and_descriptor_size_valid(self):
        for p in self.ops:
            self.assertEqual(len(p),33)
            self.assertIn(p[0],set(range(1,12))|{13})
            self.assertNotEqual(p[32],0)
            for i in range(32):
                if p[32]&(1<<i):self.assertLess(p[i],len(self.arena))

    def test_silu_fusion_retains_all_eight_layer_gates(self):
        self.assertEqual(sum(n.op_type=='VulkanSiLU' for n,a in self.nodes),48)
        self.assertEqual(sum(n.op_type=='Sigmoid' for n,a in self.nodes),0)

    def test_bias_residual_and_channel_norm_fusions(self):
        self.assertEqual(sum(n.op_type=='VulkanMatMulAdd' for n,a in self.nodes),64)
        self.assertEqual(sum(n.op_type=='VulkanChannelNorm' for n,a in self.nodes),18)

    def test_530_dispatch_layout_variant_preserves_lifetimes_and_states(self):
        self.tensors,self.nodes=ex.infer(self.model)
        self.nodes=ex.optimize(ex.live_nodes(self.model,self.tensors,self.nodes),self.tensors,layout_fusion=True)
        self.arena,self.memory=ex.allocate(self.model,self.tensors,self.nodes)
        self.ops,self.audit=ex.descriptors(self.tensors,self.nodes)
        self.assertEqual(len(self.ops),530)
        self.assertEqual(sum(n.op_type=='VulkanChannelLastConv' for n,a in self.nodes),16)
        self.assertEqual(sum(n.op_type=='Transpose' for n,a in self.nodes),4)
        self.assertEqual(sum(a.get('postSoftplus',False) for n,a in self.nodes),24)
        self.test_live_tensors_never_overlap()
        self.test_all_persistent_inputs_outputs_disjoint()
        self.test_all_graph_outputs_match_declared_shape()
        self.test_fp32_constants_copy_bitwise_without_quantization()
        self.test_every_masked_address_and_descriptor_size_valid()

    def test_broadcast_padding_and_negative_slice(self):
        self.assertEqual(ex.broadcast_strides((3,1),(2,3,5)),(0,0,0,0,1,0))
        _,shape,norm=ex.slice_spec((1,161,256),[np.array([-1]),np.array([-9223372036854775807]),np.array([1]),np.array([-1])])
        self.assertEqual(shape,(1,161,256));self.assertEqual(norm[1],(160,-1,-1))

    def test_batch_flattening_preserves_every_row_dot_and_residual_address(self):
        tensors,nodes=ex.infer(self.model)
        nodes=ex.optimize(ex.live_nodes(self.model,tensors,nodes),tensors,layout_fusion=True)
        ex.allocate(self.model,tensors,nodes)
        old,audit=ex.descriptors(tensors,nodes)
        new,new_audit=ex.descriptors(tensors,nodes,flatten_matmul_batches=True)
        self.assertEqual(audit,new_audit)
        changed=[(a,b) for a,b in zip(old,new) if a!=b]
        self.assertEqual(len(changed),40)
        for a,b in changed:
            self.assertEqual((a[6],a[9],b[6],b[9]),(1,161,161,1))
            self.assertEqual([i for i,(x,y) in enumerate(zip(a,b)) if x!=y],[6,9,10])
            for batch in range(a[9]):
                for row in range(a[6]):
                    flatrow=batch*a[6]+row
                    # All K elements retain the identical ordered addresses;
                    # shared B and the column order are unchanged.
                    self.assertEqual(a[3]+batch*a[10]+row*a[8],b[3]+flatrow*b[8])
                    self.assertEqual(a[1]+(batch*a[6]+row)*a[7],b[1]+flatrow*b[7])
                    if a[14]==2:
                        self.assertEqual(a[5]+(batch*a[6]+row)*a[7],b[5]+flatrow*b[7])
            self.assertEqual(a[32],b[32])

    def test_batch_flattening_refuses_noncontiguous_or_batched_weights(self):
        p=[0]*32;p[0]=7;p[2]=3*2*7;p[6:16]=[2,7,5,3,10,0,0,0,2,1]
        expected=p.copy();self.assertTrue(ex.flatten_broadcast_matmul(expected))
        self.assertEqual(expected[6:14],[6,7,5,1,0,0,0,0])
        for index,value in [(0,8),(2,41),(9,1),(10,11),(11,35),(12,1),(13,1),(14,3),(15,2)]:
            q=p.copy();q[index]=value;before=q.copy()
            self.assertFalse(ex.flatten_broadcast_matmul(q),(index,value))
            self.assertEqual(q,before)

    def test_concat_cache_fusion_preserves_lifetimes_and_state_banks(self):
        self.tensors,self.nodes=ex.infer(self.model)
        self.nodes=ex.optimize(ex.live_nodes(self.model,self.tensors,self.nodes),self.tensors,
                              layout_fusion=True,concat_cache_fusion=True)
        self.arena,self.memory=ex.allocate(self.model,self.tensors,self.nodes)
        self.ops,self.audit=ex.descriptors(self.tensors,self.nodes,flatten_matmul_batches=True)
        self.assertEqual(len(self.ops),466)
        concats=[p for p in self.ops if p[0]==13]
        self.assertEqual(len(concats),41);self.assertEqual(sum(p[10] for p in concats),23)
        state_outputs={(self.tensors[v.name].offset,self.tensors[v.name].count)
                       for v in self.model.graph.output if v.name.startswith('next_')}
        for p in concats:
            first,second,outer,drop,cached=p[6:11]
            self.assertEqual(p[2],(first+second)*outer)
            self.assertEqual(p[32],sum(1<<i for i in ([1,3,4,5] if cached else [1,3,4])))
            self.assertTrue(all(x==0 for x in p[11:32]))
            if cached:
                self.assertEqual(drop,second)
                self.assertIn((p[5],(first+second-drop)*outer),state_outputs)
            else:self.assertEqual((p[5],drop),(0,0))
        self.test_live_tensors_never_overlap()
        self.test_all_persistent_inputs_outputs_disjoint()
        self.test_all_graph_outputs_match_declared_shape()
        self.test_fp32_constants_copy_bitwise_without_quantization()
        self.test_every_masked_address_and_descriptor_size_valid()

    def test_every_fused_concat_and_cache_copy_is_bitwise_equivalent(self):
        t,n=ex.infer(self.model)
        old=ex.optimize(ex.live_nodes(self.model,t,n),t,layout_fusion=True)
        new=ex.fuse_concat_cache(old,t)
        before={node.output[0]:(node,a) for node,a in old}
        rng=np.random.default_rng(173)
        tested=0;cache_tested=0
        for node,a in new:
            if node.op_type!='VulkanConcat':continue
            tested+=1
            inputs=[]
            for name in node.input:
                # Include signed zero and arbitrary finite FP32 bit patterns.
                bits=rng.integers(0,0xffffffff,t[name].count,dtype=np.uint32)&np.uint32(0xfeffffff)
                bits[0]=0x80000000
                inputs.append(bits.reshape(t[name].shape))
            expected=np.concatenate(inputs,axis=a['axis'])
            first,second,outer=a['firstInner'],a['secondInner'],a['outerCount']
            linear=np.arange(expected.size,dtype=np.int64)
            group=linear//(first+second);local=linear%(first+second)
            actual=np.empty(expected.size,dtype=np.uint32);left=local<first
            actual[left]=inputs[0].reshape(-1)[group[left]*first+local[left]]
            actual[~left]=inputs[1].reshape(-1)[group[~left]*second+local[~left]-first]
            self.assertTrue(np.array_equal(actual,expected.reshape(-1)),node.output[0])
            if a['cacheFused']:
                cache_tested+=1
                crop,ca=before[node.output[1]]
                selection=tuple(slice(start,end,step) for start,end,step in ca['normalized'])
                cache_expected=expected[selection].reshape(-1)
                selected=local>=a['cacheDrop']
                cache=np.empty(cache_expected.size,dtype=np.uint32)
                destinations=group[selected]*(first+second-a['cacheDrop'])+local[selected]-a['cacheDrop']
                cache[destinations]=actual[selected]
                self.assertTrue(np.array_equal(cache,cache_expected),node.output[1])
        self.assertEqual((tested,cache_tested),(41,23))

    def test_concat_fusion_does_not_accept_noncanonical_cache_crop(self):
        t,n=ex.infer(self.model)
        n=ex.optimize(ex.live_nodes(self.model,t,n),t,layout_fusion=True)
        first_slice=next((node,a) for node,a in n if node.op_type=='Slice')
        node,attrs=first_slice
        attrs['normalized']=list(attrs['normalized']);attrs['normalized'][2]=(0,2,1)
        fused=ex.fuse_concat_cache(n,t)
        self.assertTrue(any(x.output[0]==node.output[0] and x.op_type=='Slice' for x,a in fused))
        self.assertEqual(sum(a.get('cacheFused',False) for x,a in fused),22)

    def test_conv3x3_pipeline_selection_changes_only_thirteen_hint_words(self):
        t,n=ex.infer(self.model)
        n=ex.optimize(ex.live_nodes(self.model,t,n),t,layout_fusion=True,concat_cache_fusion=True)
        ex.allocate(self.model,t,n)
        old,audit=ex.descriptors(t,n,flatten_matmul_batches=True)
        new,new_audit=ex.descriptors(t,n,flatten_matmul_batches=True,specialize_conv3x3=True)
        self.assertEqual(audit,new_audit)
        changed=[(a,b) for a,b in zip(old,new) if a!=b]
        self.assertEqual(len(changed),13)
        for a,b in changed:
            self.assertEqual(a[0],8)
            self.assertEqual([i for i,(x,y) in enumerate(zip(a,b)) if x!=y],[27])
            self.assertEqual((a[27],b[27]),(0,1))
        self.assertTrue(all(p[27]==0 for p in old if p[0]==8))

    def test_conv3x3_pipeline_rejects_every_incompatible_geometry(self):
        p=[0]*32;p[0]=8;p[6:25]=[1,64,1,161,1,128,5,161,3,3,1,1,0,1,2,1,1,1,0]
        q=p.copy();self.assertTrue(ex.mark_specialized_conv3x3(q));self.assertEqual(q[27],1)
        for index,value in [(0,7),(22,2),(24,1),(25,1),(26,1),(7,15),(8,2),
                            (14,1),(15,1),(16,2),(17,2),(18,1),(19,0),(21,2),(12,4),(9,160)]:
            q=p.copy();q[index]=value;before=q.copy()
            self.assertFalse(ex.mark_specialized_conv3x3(q),(index,value));self.assertEqual(q,before)

    def test_deterministic_export_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            a=ex.compile_plan(Path(temp)/'a')
            plan=(Path(temp)/'a/plan.bin').read_bytes()
            # The first receipt already contains its full-byte digest. Avoid
            # holding two identical 100 MiB temporary arenas on small disks.
            (Path(temp)/'a/weights.bin').unlink()
            b=ex.compile_plan(Path(temp)/'b')
            self.assertEqual(a['files'],b['files'])
            self.assertEqual(plan,(Path(temp)/'b/plan.bin').read_bytes())
            self.assertEqual(plan[:8],b'RVKPLAN1')
            version,size,ops,ni,no,ns=struct.unpack_from('<6I',plan,8)
            self.assertEqual((version,ni,no,ns),(1,33,33,31))
            self.assertEqual(len(plan),32+(ni+no)*8+ns*12+ops*132)


if __name__=='__main__':unittest.main()
