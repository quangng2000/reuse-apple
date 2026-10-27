"""Compile the pinned eight-layer FP32 graph into a fixed Vulkan execution plan.

ONNX is an offline input format only; the resulting runtime requires neither
ONNX Runtime nor an ONNX parser. Unknown operators, shapes, attributes or model
identities fail closed. Generated constants retain FP32 storage.
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import onnx
from onnx import numpy_helper

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / "tools/android-streaming-native/v2/softplus/generated/reuse-streaming-l8-native-stable-v2.onnx"
SOURCE_SHA = "db29c725c70608920e600787fcda625e98429cfbae2c2d09f1dc73726825319a"
STANDARD_SHA = "18db8d32b638702909de484d2a9cffc97ed09e6454c4caed3f0008fb0aab53e8"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strides(shape):
    return tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    constant: np.ndarray | None = None
    alias: str | None = None
    offset: int = -1

    @property
    def count(self):
        return math.prod(self.shape)


def slice_spec(shape, arrays):
    starts, ends = arrays[:2]
    axes = arrays[2] if len(arrays) > 2 else np.arange(len(starts))
    steps = arrays[3] if len(arrays) > 3 else np.ones(len(starts), dtype=np.int64)
    result = [slice(None)] * len(shape)
    for start, end, axis, step in zip(starts, ends, axes, steps):
        result[int(axis)] = slice(int(start), int(end), int(step))
    norm = [s.indices(d) for s, d in zip(result, shape)]
    return tuple(result), tuple(len(range(*s)) for s in norm), norm


def infer(model):
    tensors = {i.name: Tensor(i.name, tuple(i.dims), numpy_helper.to_array(i).copy())
               for i in model.graph.initializer}
    for i in model.graph.input:
        shape = tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
        assert all(shape), (i.name, shape)
        tensors[i.name] = Tensor(i.name, shape)
    nodes = []
    for node in model.graph.node:
        a = {v.name: onnx.helper.get_attribute_value(v) for v in node.attribute}
        ins = [tensors[x] for x in node.input if x]
        shapes = [x.shape for x in ins]
        values = [x.constant for x in ins]
        all_constant = all(x is not None for x in values)
        op = node.op_type
        result = None
        alias = None
        attrs = dict(a)
        if op == "Shape":
            result = np.array(shapes[0], dtype=np.int64)
        elif op == "ConstantOfShape":
            assert values[0] is not None
            fill = numpy_helper.to_array(a["value"]).item()
            result = np.full(tuple(values[0]), fill, dtype=np.float32)
        elif op == "Gather":
            assert all_constant
            result = np.take(values[0], values[1], axis=a.get("axis", 0))
        elif op in ("Identity", "Reshape", "Squeeze", "Unsqueeze"):
            if op == "Identity":
                shape = shapes[0]
            elif op == "Reshape":
                assert values[1] is not None
                target = [int(v) if v != 0 else shapes[0][i] for i, v in enumerate(values[1])]
                if -1 in target:
                    index = target.index(-1)
                    target[index] = math.prod(shapes[0]) // -math.prod(target)
                shape = tuple(target)
            elif op == "Squeeze":
                axes = {int(v) % len(shapes[0]) for v in values[1]}
                assert all(shapes[0][i] == 1 for i in axes)
                shape = tuple(d for i, d in enumerate(shapes[0]) if i not in axes)
            else:
                rank = len(shapes[0]) + len(values[1])
                axes = {int(v) % rank for v in values[1]}
                iterator = iter(shapes[0])
                shape = tuple(1 if i in axes else next(iterator) for i in range(rank))
            assert math.prod(shape) == math.prod(shapes[0])
            if values[0] is not None:
                result = values[0].reshape(shape)
            else:
                alias = ins[0].name
        elif op == "Transpose":
            perm = a.get("perm", list(reversed(range(len(shapes[0])))))
            shape = tuple(shapes[0][i] for i in perm)
            if all_constant:
                result = np.ascontiguousarray(values[0].transpose(perm))
            elif [i for i in perm if shapes[0][i] != 1] == [i for i in range(len(shapes[0])) if shapes[0][i] != 1]:
                alias = ins[0].name
        elif op == "Slice":
            assert all(v is not None for v in values[1:])
            selection, shape, normalized = slice_spec(shapes[0], values[1:])
            attrs["normalized"] = normalized
            if values[0] is not None:
                result = np.ascontiguousarray(values[0][selection])
        elif op == "Concat":
            axis = a["axis"] % len(shapes[0])
            shape = list(shapes[0])
            assert all(all(s[i] == shape[i] for i in range(len(shape)) if i != axis) for s in shapes)
            shape[axis] = sum(s[axis] for s in shapes)
            shape = tuple(shape)
            if all_constant:
                result = np.concatenate(values, axis=axis)
        elif op == "Pad":
            assert values[1] is not None and a.get("mode", b"constant") == b"constant"
            rank = len(shapes[0]); pads = values[1].tolist()
            shape = tuple(shapes[0][i] + pads[i] + pads[i + rank] for i in range(rank))
            attrs["pads"] = pads
            attrs["value"] = float(values[2]) if len(values) > 2 else 0.0
            assert attrs["value"] == 0.0 and all(x >= 0 for x in pads)
        elif op == "MatMul":
            assert len(shapes[0]) >= 2 and len(shapes[1]) >= 2 and shapes[0][-1] == shapes[1][-2]
            shape = np.broadcast_shapes(shapes[0][:-2], shapes[1][:-2]) + (shapes[0][-2], shapes[1][-1])
            if all_constant:
                result = np.matmul(*values)
        elif op == "Conv":
            rank = len(shapes[0]) - 2
            assert rank in (1, 2)
            pads = a.get("pads", [0] * (rank * 2)); dilation = a.get("dilations", [1] * rank); step = a.get("strides", [1] * rank)
            group = a.get("group", 1)
            assert shapes[0][1] == shapes[1][1] * group
            shape = (shapes[0][0], shapes[1][0]) + tuple((shapes[0][i+2] + pads[i] + pads[i+rank] - dilation[i]*(shapes[1][i+2]-1)-1)//step[i] + 1 for i in range(rank))
        elif op == "LayerNormalization":
            axis = a.get("axis", -1) % len(shapes[0])
            assert axis == len(shapes[0]) - 1 and len(ins) == 3 and shapes[1] == (shapes[0][-1],)
            shape = shapes[0]
        elif op in ("Neg", "Exp", "Sigmoid", "StableSoftplusV2", "Atan"):
            shape = shapes[0]
            if all_constant:
                if op == "Neg": result = -values[0]
                elif op == "Exp":
                    # ORT's CPU Exp folding is the immutable reference for A.
                    result = constant_exp(values[0])
                else: raise ValueError(f"Unexpected constant {op}")
        elif op in ("Add", "Mul", "Div", "Less", "Greater", "Equal", "PRelu"):
            shape = np.broadcast_shapes(*shapes)
            if all_constant:
                fn = {"Add": np.add, "Mul": np.multiply, "Div": np.divide,
                      "Less": np.less, "Greater": np.greater, "Equal": np.equal}.get(op)
                assert fn is not None
                result = fn(*values)
        elif op == "Where":
            shape = np.broadcast_shapes(*shapes)
        elif op in ("FrequencyScanV2", "TemporalStepV2"):
            assert a["channels"] == 256 and a["state_size"] == 16 and a["frequency_bins"] == 161 and a["abi_version"] == 2
            expected = [(1,256,16),(1,161,256),(1,161,256),(1,161,16),(1,161,16),(1,161,256),(256,16),(256,)] if op == "FrequencyScanV2" else [(161,256,16),(161,256),(161,256,1),(161,1,16),(161,1,16),(161,256),(256,16),(256,)]
            assert shapes == expected, (op, shapes)
            for name, shape in zip(node.output, [shapes[0], shapes[1]]):
                tensors[name] = Tensor(name, shape)
            nodes.append((node, attrs))
            continue
        else:
            raise ValueError(f"Unsupported operator {op}")
        if result is not None:
            shape = result.shape
        assert len(node.output) == 1 and all(d > 0 for d in shape), (op, shape)
        tensors[node.output[0]] = Tensor(node.output[0], tuple(shape), result, alias)
        if result is None and alias is None:
            nodes.append((node, attrs))
    return tensors, nodes


def constant_exp(array):
    import onnxruntime as ort
    assert ort.__version__ == '1.23.2', 'FP32 constants require the pinned reference version'
    info = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, array.shape)
    out = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, array.shape)
    graph = onnx.helper.make_graph([onnx.helper.make_node("Exp", ["x"], ["y"])], "exp", [info], [out])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(None, {"x":array})[0]


def root_name(tensors, name):
    while tensors[name].alias is not None:
        name = tensors[name].alias
    return name


def live_nodes(model, tensors, nodes):
    producer = {out: i for i, (n,a) in enumerate(nodes) for out in n.output}
    required = set()
    def visit(name):
        name = root_name(tensors, name)
        if name in producer and producer[name] not in required:
            index = producer[name]; required.add(index)
            for inp in nodes[index][0].input:
                if inp: visit(inp)
    for out in model.graph.output: visit(out.name)
    return [v for i,v in enumerate(nodes) if i in required]


def fuse_silu(nodes, tensors):
    """Eliminate materialized sigmoid tensors without changing FP32 steps."""
    users=collections.Counter(ex_input for n,a in nodes for ex_input in n.input)
    producer={value:n for n,a in nodes for value in n.output}
    removed=set();replacements={}
    for n,a in nodes:
        if n.op_type!='Mul' or len(n.input)!=2:continue
        for x,sig in (n.input, list(reversed(n.input))):
            parent=producer.get(sig)
            if parent is not None and parent.op_type=='Sigmoid' and list(parent.input)==[x] and users[sig]==1:
                replacements[n.output[0]]=onnx.helper.make_node('VulkanSiLU',[x],list(n.output))
                removed.add(sig)
                break
    return [(replacements.get(n.output[0],n),a) for n,a in nodes if n.output[0] not in removed]


def fuse_channel_norm(nodes,tensors):
    users=collections.defaultdict(list)
    producer={value:(n,a) for n,a in nodes for value in n.output}
    for n,a in nodes:
        for name in n.input:users[root_name(tensors,name)].append(n)
    removed=set();replacements={}
    for norm,attrs in nodes:
        if norm.op_type!='LayerNormalization':continue
        before=producer.get(root_name(tensors,norm.input[0]))
        if before is None or before[0].op_type!='Transpose' or before[1].get('perm')!=[0,2,3,1]:continue
        pre=before[0]
        if len(users[pre.output[0]])!=1 or len(users[norm.output[0]])!=1:continue
        post=users[norm.output[0]][0]
        if post.op_type!='Transpose' or producer[post.output[0]][1].get('perm')!=[0,3,1,2] or len(users[post.output[0]])!=1:continue
        activation=users[post.output[0]][0]
        if activation.op_type!='PRelu' or root_name(tensors,activation.input[0])!=post.output[0]:continue
        inp=tensors[pre.input[0]];slope=tensors[activation.input[1]]
        assert len(inp.shape)==4 and slope.shape==(1,inp.shape[1],1,1)
        replacement=onnx.helper.make_node('VulkanChannelNorm',[pre.input[0],norm.input[1],norm.input[2],activation.input[1]],list(activation.output))
        replacements[activation.output[0]]=(replacement,attrs)
        removed.update([pre.output[0],norm.output[0],post.output[0]])
    return [replacements.get(n.output[0],(n,a)) for n,a in nodes if n.output[0] not in removed]


def fuse_matmul_add(nodes,tensors):
    users=collections.defaultdict(list)
    for n,a in nodes:
        for name in n.input:users[root_name(tensors,name)].append(n)
    removed=set();replacements={}
    for mul,attrs in nodes:
        if mul.op_type!='MatMul' or len(users[mul.output[0]])!=1:continue
        add=users[mul.output[0]][0]
        if add.op_type!='Add':continue
        others=[name for name in add.input if root_name(tensors,name)!=mul.output[0]]
        if len(others)!=1:continue
        other=others[0];shape=tensors[other].shape;outshape=tensors[add.output[0]].shape
        if shape==(outshape[-1],):mode=1
        elif shape==outshape:mode=2
        else:continue
        replacement=onnx.helper.make_node('VulkanMatMulAdd',list(mul.input)+[other],list(add.output))
        replacements[add.output[0]]=(replacement,dict(attrs,addMode=mode))
        removed.add(mul.output[0])
    return [replacements.get(n.output[0],(n,a)) for n,a in nodes if n.output[0] not in removed]


def remove_inverse_transposes(nodes,tensors):
    users=collections.defaultdict(list)
    producer={value:(n,a) for n,a in nodes for value in n.output}
    for n,a in nodes:
        for name in n.input:users[root_name(tensors,name)].append(n)
    removed=set()
    for second,attrs in nodes:
        if second.op_type!='Transpose':continue
        first_pair=producer.get(root_name(tensors,second.input[0]))
        if first_pair is None:continue
        first,fa=first_pair
        if first.op_type!='Transpose' or len(users[first.output[0]])!=1 or first.output[0] in removed:continue
        source=tensors[first.input[0]]
        indices=np.arange(source.count,dtype=np.int32).reshape(source.shape)
        mapped=indices.transpose(fa['perm']).reshape(tensors[second.input[0]].shape).transpose(attrs['perm'])
        if np.array_equal(mapped.reshape(-1),np.arange(source.count,dtype=np.int32)):
            assert not second.output[0].startswith(('next_','enhanced_'))
            tensors[second.output[0]].alias=first.input[0]
            removed.update((first.output[0],second.output[0]))
    return [(n,a) for n,a in nodes if n.output[0] not in removed]


def fuse_channel_last_depthwise(nodes,tensors):
    users=collections.defaultdict(list)
    producer={value:(n,a) for n,a in nodes for value in n.output}
    for n,a in nodes:
        for name in n.input:users[root_name(tensors,name)].append(n)
    removed=set();replacements={}
    for conv,attrs in nodes:
        if conv.op_type!='Conv' or attrs.get('group')!=256:continue
        pair=producer.get(root_name(tensors,conv.input[0]))
        if pair is None or pair[0].op_type!='Transpose' or pair[1].get('perm')!=[0,2,1]:continue
        pre=pair[0]
        if len(users[pre.output[0]])!=1 or len(users[conv.output[0]])!=1:continue
        crop=users[conv.output[0]][0]
        if crop.op_type!='Slice' or len(users[crop.output[0]])!=1:continue
        crop_attrs=producer[crop.output[0]][1]
        if crop_attrs.get('normalized')!=[(0,1,1),(0,256,1),(0,161,1)]:continue
        post=users[crop.output[0]][0]
        if post.op_type!='Transpose' or producer[post.output[0]][1].get('perm')!=[0,2,1] or len(users[post.output[0]])!=1:continue
        activation=users[post.output[0]][0]
        if activation.op_type!='VulkanSiLU':continue
        assert tensors[pre.input[0]].shape==(1,161,256) and tensors[crop.output[0]].shape==(1,256,161)
        replacement=onnx.helper.make_node('VulkanChannelLastConv',[pre.input[0]]+list(conv.input[1:]),list(activation.output))
        replacements[activation.output[0]]=(replacement,attrs)
        removed.update((pre.output[0],conv.output[0],crop.output[0],post.output[0]))
    return [replacements.get(n.output[0],(n,a)) for n,a in nodes if n.output[0] not in removed]


def fuse_matmul_softplus(nodes,tensors):
    users=collections.defaultdict(list)
    for n,a in nodes:
        for name in n.input:users[root_name(tensors,name)].append(n)
    removed=set();replacements={}
    for mul,attrs in nodes:
        if mul.op_type!='VulkanMatMulAdd' or len(users[mul.output[0]])!=1:continue
        activation=users[mul.output[0]][0]
        if activation.op_type!='StableSoftplusV2':continue
        replacement=onnx.helper.make_node('VulkanMatMulAdd',list(mul.input),list(activation.output))
        replacements[activation.output[0]]=(replacement,dict(attrs,postSoftplus=True))
        removed.add(mul.output[0])
    return [replacements.get(n.output[0],(n,a)) for n,a in nodes if n.output[0] not in removed]


def fuse_concat_cache(nodes, tensors):
    """Copy each concatenation and its exact causal next cache in one dispatch.

    Both outputs keep dedicated storage. The state output contains the same
    FP32 bit patterns as Slice(Concat(old_cache, current), drop=current_size),
    without first reading back the materialized concatenation. No arithmetic,
    strided tensor view or recurrent bank alias is introduced.
    """
    users=collections.defaultdict(list)
    attrs_by_output={n.output[0]:a for n,a in nodes}
    persistent_outputs={root_name(tensors,name) for name in tensors if name.startswith('next_')}
    for n,a in nodes:
        for name in n.input:
            if name:users[root_name(tensors,name)].append(n)
    removed=set();replacements={}
    for n,a in nodes:
        if n.op_type!='Concat' or len(n.input)!=2:continue
        inputs=[tensors[name] for name in n.input];out=tensors[n.output[0]]
        axis=a['axis']%len(out.shape)
        inner=math.prod(out.shape[axis+1:]);outer=math.prod(out.shape[:axis])
        first=inputs[0].shape[axis]*inner;second=inputs[1].shape[axis]*inner
        assert all(t.count==outer*span for t,span in zip(inputs,(first,second)))
        assert out.count==outer*(first+second) and min(first,second,outer)>0
        outputs=list(n.output);drop=0
        if root_name(tensors,n.input[0]).startswith('cache_'):
            candidates=[]
            for crop in users[n.output[0]]:
                if crop.op_type!='Slice' or crop.output[0] not in persistent_outputs:continue
                normalized=attrs_by_output[crop.output[0]].get('normalized')
                expected=[(0,dim,1) for dim in out.shape]
                expected[axis]=(inputs[1].shape[axis],out.shape[axis],1)
                if normalized==expected and tensors[crop.output[0]].shape==inputs[0].shape:
                    candidates.append(crop)
            assert len(candidates)<=1, 'Ambiguous causal cache output'
            if candidates:
                crop=candidates[0];outputs.append(crop.output[0]);drop=second
                removed.add(crop.output[0])
        replacement=onnx.helper.make_node('VulkanConcat',list(n.input),outputs)
        replacements[n.output[0]]=(replacement,{'firstInner':first,'secondInner':second,
                    'outerCount':outer,'cacheDrop':drop,'cacheFused':len(outputs)==2,'axis':axis})
    return [replacements.get(n.output[0],(n,a)) for n,a in nodes if n.output[0] not in removed]


def optimize(nodes,tensors,layout_fusion=False,concat_cache_fusion=False):
    nodes=fuse_matmul_add(fuse_channel_norm(fuse_silu(nodes,tensors),tensors),tensors)
    if layout_fusion:
        nodes=fuse_matmul_softplus(fuse_channel_last_depthwise(remove_inverse_transposes(nodes,tensors),tensors),tensors)
    if concat_cache_fusion:nodes=fuse_concat_cache(nodes,tensors)
    return nodes


def allocate(model, tensors, nodes):
    """Separate immutable/persistent storage from liveness-reused scratch."""
    align = lambda n: (n + 63) // 64 * 64
    persistent = list(dict.fromkeys(root_name(tensors, v.name) for v in list(model.graph.input) + list(model.graph.output)))
    used = {root_name(tensors, name) for n,a in nodes for name in n.input if name}
    constants = sorted(name for name in used if tensors[name].constant is not None)
    cursor = 64
    for name in constants + persistent:
        t = tensors[name]
        if t.offset >= 0: continue
        t.offset = cursor
        cursor += align(t.count)
    scratch_start = cursor
    last = {}
    for i, (n,a) in enumerate(nodes):
        for name in n.input:
            if name: last[root_name(tensors, name)] = i
        for name in n.output:
            last.setdefault(root_name(tensors, name), i)
    free = []
    active = []
    for i,(node,attrs) in enumerate(nodes):
        for name in node.output:
            t = tensors[name]
            assert t.alias is None and t.constant is None
            if t.offset >= 0: continue
            count = align(t.count)
            fits = [(size,j) for j,(off,size) in enumerate(free) if size >= count]
            if fits:
                _, j = min(fits); offset,size = free.pop(j)
                t.offset = offset
                if size > count: free.append((offset+count,size-count))
            else:
                t.offset = cursor; cursor += count
            active.append((name,t.offset,count))
        keep = []
        for name,off,size in active:
            if last.get(name,i) <= i: free.append((off,size))
            else: keep.append((name,off,size))
        active = keep
        free.sort()
        merged = []
        for off,size in free:
            if merged and merged[-1][0]+merged[-1][1] == off:
                oldoff,oldsize=merged[-1];merged[-1]=(oldoff,oldsize+size)
            else: merged.append((off,size))
        free=merged
    for t in tensors.values():
        if t.alias is not None:
            t.offset=tensors[root_name(tensors,t.name)].offset
    arena=np.zeros(cursor,dtype='<f4')
    for name in constants:
        t=tensors[name]
        assert t.constant is not None
        arena[t.offset:t.offset+t.count]=t.constant.reshape(-1).astype(np.float32)
    return arena,{"persistentFloats":scratch_start-64,"scratchFloats":cursor-scratch_start,"alignmentFloats":64}


def pad6(shape):
    assert len(shape) <= 6
    return (1,)*(6-len(shape)) + tuple(shape)


def broadcast_strides(shape, target):
    shape=pad6(shape); target=pad6(target)
    return tuple(0 if d==1 else s for d,s in zip(shape,strides(shape)))


def fbits(x): return struct.unpack('<I',struct.pack('<f',x))[0]


def flatten_broadcast_matmul(p):
    """Fold contiguous A batches into rows without changing any dot product.

    The fixed model's temporal projections have shape [161, 1, K]. Treating
    them as 161 independent one-row matrices wastes almost every tiled row.
    With shared B and contiguous A/output, [161, K] addresses exactly the same
    values in the same K order. Both column bias and full-output residuals
    retain their original linear addresses. Unknown descriptor modes stay put.
    """
    if (p[0] != 7 or p[9] <= 1 or p[12] != 0 or p[13] != 0
            or p[10] != p[6] * p[8] or p[11] != 0
            or p[2] != p[9] * p[6] * p[7]
            or p[14] not in (0, 1, 2) or p[15] not in (0, 1)):
        return False
    p[6] *= p[9]
    p[9] = 1
    p[10] = 0
    return True


def mark_specialized_conv3x3(p):
    """Select an equivalent dense 3x3 pipeline only for its exact geometry."""
    if (p[0]!=8 or p[22]!=1 or p[24]!=0 or p[25]!=0 or p[26]!=0
            or p[7]<16 or p[8]!=1 or p[14]!=3 or p[15]!=3
            or p[16]!=1 or p[17]!=1 or p[18]!=0 or p[19]!=1
            or p[21]!=1 or p[12]!=1+2*p[20] or p[9]!=p[13]):
        return False
    p[27]=1
    return True


def descriptors(tensors, nodes, flatten_matmul_batches=False, specialize_conv3x3=False):
    records=[]; audit=[]
    unary={"Neg":1,"Exp":2,"Sigmoid":3,"StableSoftplusV2":4,"Atan":5,"VulkanSiLU":14}
    binary={"Add":1,"Mul":2,"Div":3,"PRelu":4,"Less":5,"Greater":6,"Equal":7}
    def emit(p, mask, node, suffix=''):
        assert len(p)==32
        assert all(-(2**31)<=v<2**32 for v in p)
        records.append([int(x)&0xffffffff for x in p]+[sum(1<<i for i in mask)])
        audit.append({"index":len(records)-1,"operator":node.op_type,"output":node.output[0]+suffix,"inputs":list(node.input)})
    for n,a in nodes:
        ins=[tensors[x] for x in n.input if x]
        out=tensors[n.output[0]]
        p=[0]*32;p[1]=out.offset;p[2]=out.count
        for j,t in enumerate(ins[:3]):p[3+j]=t.offset
        mask={1,3}
        op=n.op_type
        if op in unary:
            p[0]=1;p[6]=unary[op]
        elif op in binary:
            p[0]=2;p[6]=binary[op]
            p[8:14]=pad6(out.shape);p[14:20]=broadcast_strides(ins[0].shape,out.shape);p[20:26]=broadcast_strides(ins[1].shape,out.shape)
            mask.add(4)
        elif op=="Where":
            p[0]=3;p[8:14]=pad6(out.shape)
            for i,t in enumerate(ins):p[14+6*i:20+6*i]=broadcast_strides(t.shape,out.shape)
            mask.update((4,5))
        elif op=="Transpose":
            p[0]=4;perm=a.get("perm",list(reversed(range(len(ins[0].shape)))))
            p[8:14]=pad6(out.shape);s=strides(ins[0].shape)
            p[14:20]=(0,)*(6-len(perm))+tuple(s[i] for i in perm)
        elif op=="Slice":
            p[0]=4;s=strides(ins[0].shape);norm=a["normalized"]
            p[3]+=sum(s[i]*x[0] for i,x in enumerate(norm))
            p[8:14]=pad6(out.shape)
            p[14:20]=(0,)*(6-len(norm))+tuple(s[i]*x[2] for i,x in enumerate(norm))
        elif op=="Pad":
            p[0]=5;p[6]=fbits(a["value"]);p[8:14]=pad6(out.shape);p[14:20]=pad6(ins[0].shape)
            rank=len(out.shape);p[20:26]=(0,)*(6-rank)+tuple(a["pads"][:rank])
        elif op=="Concat":
            axis=a["axis"]%len(out.shape);cursor=0;os=strides(out.shape)
            for i,t in enumerate(ins):
                q=[0]*32;q[0]=6;q[1]=out.offset+cursor*os[axis];q[2]=t.count;q[3]=t.offset
                q[8:14]=pad6(t.shape);q[14:20]=(0,)*(6-len(os))+os
                emit(q,{1,3},n,f":part{i}");cursor+=t.shape[axis]
            continue
        elif op=="VulkanConcat":
            p[0]=13;p[5]=tensors[n.output[1]].offset if a['cacheFused'] else 0
            p[6:11]=[a['firstInner'],a['secondInner'],a['outerCount'],a['cacheDrop'],int(a['cacheFused'])]
            mask.add(4)
            if a['cacheFused']:mask.add(5)
        elif op in ("MatMul","VulkanMatMulAdd"):
            p[0]=7
            m,k=ins[0].shape[-2:];k2,nn=ins[1].shape[-2:];assert k==k2
            batch=math.prod(out.shape[:-2]);abatch=math.prod(ins[0].shape[:-2]);bbatch=math.prod(ins[1].shape[:-2])
            assert abatch in (1,batch) and bbatch in (1,batch)
            p[6:14]=[m,nn,k,batch,0 if abatch==1 else m*k,0 if bbatch==1 else k*nn,0,0]
            mask.add(4)
            if op=='VulkanMatMulAdd':p[14]=a['addMode'];mask.add(5)
            p[15]=int(a.get('postSoftplus',False))
            if flatten_matmul_batches:flatten_broadcast_matmul(p)
        elif op in ("Conv","VulkanChannelLastConv"):
            p[0]=8;rank=len(ins[0].shape)-2;assert rank in (1,2)
            x=ins[0].shape;w=ins[1].shape;y=out.shape
            if op=='VulkanChannelLastConv':
                x=(x[0],x[2],1,x[1]);w=w[:2]+(1,w[2]);y=(y[0],y[2],1,y[1])
                p[25]=1;p[26]=1
            elif rank==1:x=x[:2]+(1,x[2]);w=w[:2]+(1,w[2]);y=y[:2]+(1,y[2])
            dil=a.get("dilations",[1]*rank);steps=a.get("strides",[1]*rank);pads=a.get("pads",[0]*(2*rank))
            if rank==1:dil=[1]+dil;steps=[1]+steps;pads=[0,pads[0],0,pads[1]]
            p[6:25]=[y[0],y[1],y[2],y[3],x[0],x[1],x[2],x[3],w[2],w[3],steps[0],steps[1],pads[0],pads[1],dil[0],dil[1],a.get("group",1),int(len(ins)>2),0]
            mask.add(4)
            if len(ins)>2:mask.add(5)
            if specialize_conv3x3:mark_specialized_conv3x3(p)
        elif op in ("LayerNormalization","VulkanChannelNorm"):
            p[0]=9;p[6]=math.prod(out.shape[:-1]);p[7]=out.shape[-1];p[8]=fbits(a.get("epsilon",1e-5));p[9]=1;p[10]=1
            mask.update((4,5))
            if op=='VulkanChannelNorm':
                nn,channels,h,w=out.shape
                p[6]=nn*h*w;p[7]=channels;p[11]=1;p[12]=h*w;p[13]=1;p[14]=ins[3].offset;p[15]=ins[3].count
                mask.add(14)
        elif op in ("TemporalStepV2","FrequencyScanV2"):
            p[0]=11 if op=="TemporalStepV2" else 10
            p[1]=tensors[n.output[1]].offset;p[2]=tensors[n.output[1]].count
            for i,t in enumerate(ins):p[3+i]=t.offset
            p[11]=out.offset;p[12:15]=[161,256,16]
            mask=set(range(3,12))|{1}
        else:raise ValueError(op)
        emit(p,mask,n)
    return records,audit


def compile_plan(output,layout_fusion=False,flatten_matmul_batches=False,concat_cache_fusion=False,
                 specialize_conv3x3=False):
    assert digest(SOURCE)==SOURCE_SHA
    model=onnx.load(SOURCE)
    before={i.name:hashlib.sha256(i.SerializeToString()).hexdigest() for i in model.graph.initializer}
    tensors,nodes=infer(model)
    nodes=optimize(live_nodes(model,tensors,nodes),tensors,layout_fusion=layout_fusion,concat_cache_fusion=concat_cache_fusion)
    arena,memory=allocate(model,tensors,nodes)
    records,audit=descriptors(tensors,nodes,flatten_matmul_batches=flatten_matmul_batches,
                              specialize_conv3x3=specialize_conv3x3)
    unflattened,_=descriptors(tensors,nodes,specialize_conv3x3=specialize_conv3x3) if flatten_matmul_batches else (records,audit)
    flattened_count=sum(a != b for a,b in zip(records,unflattened))
    inputs=[(tensors[v.name].offset,tensors[v.name].count) for v in model.graph.input]
    outputs=[(tensors[v.name].offset,tensors[v.name].count) for v in model.graph.output]
    assert len(inputs)==len(outputs)==33
    names=[v.name for v in model.graph.output]
    pairs=[]
    for v in list(model.graph.input)[2:]:
        target='next_'+v.name[len('cache_'):]
        t=tensors[v.name];o=tensors[target]
        assert t.count==o.count and target in names and t.offset!=o.offset
        pairs.append((t.offset,o.offset,t.count))
    spans=sorted((off,off+count) for pair in pairs for off,count in [(pair[0],pair[2]),(pair[1],pair[2])])
    assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
    assert len(arena)*4<=128*1024**2, f'arena exceeds Vulkan guaranteed128MiB range: {len(arena)*4}'
    header=b'RVKPLAN1'+struct.pack('<6I',1,len(arena),len(records),len(inputs),len(outputs),len(pairs))
    payload=header+b''.join(struct.pack('<2I',*r) for r in inputs+outputs)+b''.join(struct.pack('<3I',*r) for r in pairs)+b''.join(struct.pack('<33I',*r) for r in records)
    output.mkdir(parents=True,exist_ok=True)
    (output/'plan.bin').write_bytes(payload);(output/'weights.bin').write_bytes(arena.tobytes())
    receipt={
        'schemaVersion':1,'executionBackend':'native-vulkan-fp32','layers':8,'sampleRate':16000,'hop':320,'fft':640,'stateCount':31,
        'checkpointSha256':'6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade',
        'baselineModelSha256':STANDARD_SHA,'sourceModelSha256':SOURCE_SHA,'selectedWeightBytesUnchanged':True,
        'precision':'float32','recordWords':33,'arenaFloats':len(arena),'arenaBytes':len(arena)*4,'memory':memory,
        'runtimeNodeCount':len(nodes),'dispatchCount':len(records),'operators':dict(collections.Counter(n.op_type for n,a in nodes)),
        'layoutFusion':layout_fusion,
        'flattenMatmulBatches':flatten_matmul_batches,'flattenedMatmulCount':flattened_count,
        'concatCacheFusion':concat_cache_fusion,
        'fusedConcatCount':sum(n.op_type=='VulkanConcat' for n,a in nodes),
        'fusedCacheCount':sum(n.op_type=='VulkanConcat' and a['cacheFused'] for n,a in nodes),
        'specializedConv3x3':specialize_conv3x3,
        'specializedConv3x3Count':sum(p[0]==8 and p[27]==1 for p in records),
        'inputs':[{'name':v.name,'offset':tensors[v.name].offset,'count':tensors[v.name].count,'shape':tensors[v.name].shape} for v in model.graph.input],
        'outputs':[{'name':v.name,'offset':tensors[v.name].offset,'count':tensors[v.name].count,'shape':tensors[v.name].shape} for v in model.graph.output],
        'statePairs':pairs,'sourceInitializerSha256':before,'constantEvaluation':'FP32; Exp uses ONNX Runtime1.23.2CPU offline only',
        'files':{n:{'bytes':(output/n).stat().st_size,'sha256':digest(output/n)} for n in ['plan.bin','weights.bin']},
        'dispatches':audit,
        'tensors':{name:{'shape':t.shape,'offset':t.offset,'count':t.count,'alias':t.alias,'constant':t.constant is not None} for name,t in tensors.items() if t.offset>=0},
    }
    assert before=={i.name:hashlib.sha256(i.SerializeToString()).hexdigest() for i in model.graph.initializer}
    (output/'export.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "generated")
    parser.add_argument('--layout-fusion',action='store_true')
    parser.add_argument('--flatten-matmul-batches',action='store_true')
    parser.add_argument('--fuse-concat-cache',action='store_true')
    parser.add_argument('--specialize-conv3x3',action='store_true')
    args = parser.parse_args()
    receipt=compile_plan(args.output,layout_fusion=args.layout_fusion,flatten_matmul_batches=args.flatten_matmul_batches,
                         concat_cache_fusion=args.fuse_concat_cache,specialize_conv3x3=args.specialize_conv3x3)
    print(json.dumps({k:receipt[k] for k in ['runtimeNodeCount','dispatchCount','arenaBytes','memory','operators','files']},indent=2))


if __name__ == "__main__": main()
