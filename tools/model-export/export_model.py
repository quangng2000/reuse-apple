"""Reproducible spectral ONNX export; standard operators, no CUDA/custom ops.

See reference.py for NVIDIA/Mamba source attribution and checkpoint identity.
The ONNX graph retains all 30 blocks and 120 sequential Mamba scans in FP32.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
from reference import CHECKPOINT_SHA256, SOURCE_REVISION, DEFAULT_CHECKPOINT, verified_weights


class Graph:
    def __init__(self, weights):
        self.nodes, self.initializers = [], []
        self.count = 0
        self.constants = {}
        self.weights = {k: v.numpy() for k, v in weights.items()}
        for k, v in self.weights.items():
            self.initializers.append(nh.from_array(v, k))

    def name(self, stem):
        self.count += 1
        return f"{stem}_{self.count}"

    def op(self, kind, *inputs, **attrs):
        out = self.name(kind.lower())
        self.nodes.append(h.make_node(kind, list(inputs), [out], **attrs))
        return out

    def const(self, value, dtype=np.int64):
        a = np.asarray(value, dtype=dtype)
        key = (a.dtype.str, a.shape, a.tobytes())
        if key not in self.constants:
            n = self.name("constant")
            self.initializers.append(nh.from_array(a, n))
            self.constants[key] = n
        return self.constants[key]

    def shape(self, x, axis=None):
        s = self.op("Shape", x)
        return s if axis is None else self.op("Gather", s, self.const([axis]), axis=0)

    def cat(self, *xs, axis=0):
        return self.op("Concat", *xs, axis=axis)

    def reshape(self, x, shape):
        return self.op("Reshape", x, self.const(shape) if isinstance(shape, (list, tuple)) else shape)

    def transpose(self, x, perm):
        return self.op("Transpose", x, perm=perm)

    def unsqueeze(self, x, axes):
        return self.op("Unsqueeze", x, self.const(axes))

    def slice(self, x, lo, hi, axis, step=1):
        a = self.const([lo]) if isinstance(lo, int) else lo
        b = self.const([hi]) if isinstance(hi, int) else hi
        return self.op("Slice", x, a, b, self.const([axis]), self.const([step]))

    def linear(self, x, p, bias=True):
        weight = self.transpose(p + ".weight", [1, 0])
        y = self.op("MatMul", x, weight)
        return self.op("Add", y, p + ".bias") if bias and p + ".bias" in self.weights else y

    def conv(self, x, p, padding=(0, 0), stride=(1, 1), dilation=(1, 1)):
        return self.op("Conv", x, p + ".weight", p + ".bias", pads=[*padding, *padding],
                       strides=list(stride), dilations=list(dilation))

    def norm_prelu(self, x, p):
        x = self.op("InstanceNormalization", x, p + ".1.weight", p + ".1.bias", epsilon=1e-5)
        slope = self.reshape(p + ".2.weight", [1, 64, 1, 1])
        return self.op("PRelu", x, slope)

    def dense(self, x, p):
        skip = x
        for i in range(4):
            q = f"{p}.dense_block.{i}"
            x = self.conv(skip, q + ".0", padding=(1 << i, 1), dilation=(1 << i, 1))
            x = self.norm_prelu(x, q)
            if i < 3:
                skip = self.cat(x, skip, axis=1)
        return x

    def up(self, x, p, ratio):
        x = self.op("Pad", x, self.const([0, 0, 0, 1, 0, 0, 0, 1]), mode="constant")
        x = self.conv(x, p + ".0.conv")
        batch, rows, cols = self.shape(x, 0), self.shape(x, 2), self.shape(x, 3)
        x = self.reshape(x, self.cat(batch, self.const([ratio, 64]), rows, cols))
        x = self.transpose(x, [0, 2, 3, 4, 1])
        x = self.reshape(x, self.cat(batch, self.const([64]), rows,
                                     self.op("Mul", cols, self.const([ratio]))))
        return self.norm_prelu(x, p)

    def decoder(self, x, p):
        x = self.dense(x, p + ".dense_block")
        x = self.up(x, p + ".up_conv1", 2)
        return self.transpose(self.up(self.transpose(x, [0, 1, 3, 2]), p + ".up_conv2", 4), [0, 1, 3, 2])

    def scan(self, u, dt, b, c, z, p):
        # Scan body captures constant coefficients from its parent graph. Its one
        # loop-carried tensor is exactly [batch,256,16], initialized once to zero.
        a = self.op("Neg", self.op("Exp", p + ".A_log"))
        dt = self.op("Softplus", self.op("Add", dt, p + ".dt_proj.bias"))
        zero_shape = self.cat(self.shape(u, 0), self.const([256, 16]))
        zero = self.op("ConstantOfShape", zero_shape, value=nh.from_array(np.array([0], dtype=np.float32)))
        body_nodes = []
        idx = 0

        def op(kind, *inputs, **attrs):
            nonlocal idx
            idx += 1
            out = f"{p}_body_{idx}"
            body_nodes.append(h.make_node(kind, list(inputs), [out], **attrs))
            return out

        axes2 = self.const([2])
        axes1 = self.const([1])
        d = op("Unsqueeze", "delta_t", axes2)
        uu = op("Unsqueeze", "u_t", axes2)
        bb = op("Unsqueeze", "b_t", axes1)
        cc = op("Unsqueeze", "c_t", axes1)
        decay = op("Exp", op("Mul", d, a))
        update = op("Mul", op("Mul", d, bb), uu)
        state = op("Add", op("Mul", decay, "state_in"), update)
        y = op("ReduceSum", op("Mul", state, cc), axes2, keepdims=0)
        skip = op("Mul", p + ".D", "u_t")
        gate = op("Mul", "z_t", op("Sigmoid", "z_t"))
        out = op("Mul", op("Add", y, skip), gate)
        inputs = [h.make_tensor_value_info("state_in", T.FLOAT, [None, 256, 16])]
        for n, channels in [("u_t", 256), ("delta_t", 256), ("b_t", 16), ("c_t", 16), ("z_t", 256)]:
            inputs.append(h.make_tensor_value_info(n, T.FLOAT, [None, channels]))
        outputs = [h.make_tensor_value_info(state, T.FLOAT, [None, 256, 16]),
                   h.make_tensor_value_info(out, T.FLOAT, [None, 256])]
        body = h.make_graph(body_nodes, p + ".recurrence", inputs, outputs)
        final, values = self.name("state_final"), self.name("scan_values")
        self.nodes.append(h.make_node("Scan", [zero, u, dt, b, c, z], [final, values],
                                      name=p + ".scan", body=body, num_scan_inputs=5,
                                      scan_input_axes=[1] * 5, scan_output_axes=[1]))
        return values

    def mamba(self, x, p):
        length = self.shape(x, 1)
        xz = self.linear(x, p + ".in_proj")
        u, z = self.slice(xz, 0, 256, 2), self.slice(xz, 256, 512, 2)
        u = self.transpose(u, [0, 2, 1])
        u = self.op("Conv", u, p + ".conv1d.weight", p + ".conv1d.bias", group=256, pads=[3, 3])
        u = self.transpose(self.slice(u, 0, length, 2), [0, 2, 1])
        u = self.op("Mul", u, self.op("Sigmoid", u))
        proj = self.linear(u, p + ".x_proj", False)
        dt = self.linear(self.slice(proj, 0, 4, 2), p + ".dt_proj", False)
        b, c = self.slice(proj, 4, 20, 2), self.slice(proj, 20, 36, 2)
        return self.linear(self.scan(u, dt, b, c, z, p), p + ".out_proj")

    def reverse(self, x):
        return self.slice(x, -1, np.iinfo(np.int64).min, 1, -1)

    def bidirectional(self, x, p):
        fw = self.op("Add", self.mamba(x, p + ".forward_blocks"), x)
        rev = self.reverse(x)
        bw = self.reverse(self.op("Add", self.mamba(rev, p + ".backward_blocks"), rev))
        x = self.linear(self.cat(fw, bw, axis=2), p + ".output_proj")
        return self.op("LayerNormalization", x, p + ".norm.weight", p + ".norm.bias", axis=-1, epsilon=1e-5)

    def atan2(self, y, x):
        # ONNX has Atan but no Atan2. Explicit quadrants also handle both-zero
        # samples without exposing the NaN of 0/0 at the graph output.
        zero, pi = self.const(0.0, np.float32), self.const(np.pi, np.float32)
        one = self.const(1.0, np.float32)
        halfpi = self.const(np.pi / 2, np.float32)
        angle = self.op("Atan", self.op("Div", y, x))
        # Reciprocal preserves the sign of IEEE signed zero (±infinity).
        xneg = self.op("Less", self.op("Div", one, x), zero)
        yneg = self.op("Less", self.op("Div", one, y), zero)
        correction = self.op("Where", yneg, self.op("Neg", pi), pi)
        angle = self.op("Where", self.op("Less", x, zero), self.op("Add", angle, correction), angle)
        bothzero = self.op("Where", xneg, correction, y)
        atzero = self.op("Where", self.op("Greater", y, zero), halfpi,
                         self.op("Where", self.op("Less", y, zero), self.op("Neg", halfpi), bothzero))
        return self.op("Where", self.op("Equal", x, zero), atzero, angle)

    def build(self, *, inspect=False):
        frames = self.shape("magnitude", 2)
        x = self.cat(self.unsqueeze(self.transpose("magnitude", [0, 2, 1]), [1]),
                     self.unsqueeze(self.transpose("phase", [0, 2, 1]), [1]), axis=1)
        x = self.op("Pad", x, self.const([0, 0, 0, 0, 0, 0, 2, 2]), mode="constant")
        x = self.norm_prelu(self.conv(x, "dense_encoder.dense_conv_1.0"), "dense_encoder.dense_conv_1")
        x = self.dense(x, "dense_encoder.dense_block")
        x = self.norm_prelu(self.conv(x, "dense_encoder.dense_conv_2.0", stride=(4, 2)), "dense_encoder.dense_conv_2")
        inspection = [("encoder", x)] if inspect else []
        for i in range(30):
            nt = self.shape(x, 2)
            tx = self.reshape(self.transpose(x, [0, 3, 2, 1]), [161, -1, 64])
            tx = self.op("Add", self.bidirectional(tx, f"TSMamba.{i}.time_mamba"), tx)
            tx = self.reshape(tx, self.cat(self.const([1, 161]), nt, self.const([64])))
            fx = self.reshape(self.transpose(tx, [0, 2, 1, 3]), [-1, 161, 64])
            fx = self.op("Add", self.bidirectional(fx, f"TSMamba.{i}.freq_mamba"), fx)
            x = self.transpose(self.reshape(fx, self.cat(self.const([1]), nt, self.const([161, 64]))), [0, 3, 1, 2])
            if inspect and i in (0, 29):
                inspection.append((f"block{i+1}", x))
        mag = self.conv(self.decoder(x, "mask_decoder"), "mask_decoder.final_conv")
        features = self.decoder(x, "phase_decoder")
        real, imag = self.conv(features, "phase_decoder.phase_conv_r"), self.conv(features, "phase_decoder.phase_conv_i")
        pha = self.atan2(imag, real)
        outputs = []
        for src, name in [(mag, "enhanced_magnitude"), (pha, "enhanced_phase")]:
            src = self.slice(self.slice(src, 0, frames, 2), 0, 321, 3)
            src = self.transpose(self.op("Squeeze", src, self.const([1])), [0, 2, 1])
            self.nodes.append(h.make_node("Identity", [src], [name]))
            outputs.append(h.make_tensor_value_info(name, T.FLOAT, [1, 321, "frames"]))
        for name, src in inspection:
            self.nodes.append(h.make_node("Identity", [src], [name]))
            outputs.append(h.make_tensor_value_info(name, T.FLOAT, [1, 64, "encoded_frames", 161]))
        graph = h.make_graph(self.nodes, "RE-USE 30-layer commercial FP32 spectral model",
                             [h.make_tensor_value_info(n, T.FLOAT, [1, 321, "frames"]) for n in ("magnitude", "phase")],
                             outputs, self.initializers)
        model = h.make_model(graph, producer_name="reuse-audio reproducible spectral exporter",
                             opset_imports=[h.make_opsetid("", 17)])
        model.ir_version = 9
        h.set_model_props(model, {"checkpoint_sha256": CHECKPOINT_SHA256, "source_revision": SOURCE_REVISION,
                                  "precision": "float32", "num_layers": "30", "mamba_scans": "120"})
        onnx.checker.check_model(model, full_check=True)
        return model


def manifest(path):
    return {
        "schemaVersion": 1, "modelId": "nvidia/reuse-30-fp32", "modelFile": path.name,
        "modelSha256": hashlib.sha256(path.read_bytes()).hexdigest(), "modelBytes": path.stat().st_size,
        "checkpointSha256": CHECKPOINT_SHA256, "sourceRevision": SOURCE_REVISION,
        "numLayers": 30, "parameterCount": 9_607_747, "precision": "float32",
        "sampleRate": 16000, "fftSize": 640, "hopSize": 80,
        "windowSamples": 64000, "overlapSamples": 8000,
        "inputNames": ["magnitude", "phase"], "outputNames": ["enhanced_magnitude", "enhanced_phase"],
        "tensorShape": [1, 321, "frames"], "mambaScans": 120, "onnxOpset": 17,
        "onnxRuntimeVersion": "1.23.2", "license": "OpenMDW-1.1",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "generated/reuse-30-fp32.onnx")
    p.add_argument("--inspect", action="store_true", help="Extra encoder/block1/block30 outputs for parity debugging")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    model = Graph(verified_weights(a.checkpoint)).build(inspect=a.inspect)
    onnx.save_model(model, a.output, save_as_external_data=False)
    info = manifest(a.output)
    a.output.with_suffix(".manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
