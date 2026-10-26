"""Stateful one-hop NVIDIA Real-time RE-USE export. NVIDIA NSCLv1 only.

Architecture derived from the pinned NVIDIA reference and its Swift research
port. Copyright(c)2025 NVIDIA CORPORATION. Retain LICENSE with redistribution.
Mamba recurrence: Tri Dao/Albert Gu, Apache-2.0 (MAMBA-LICENSE.txt).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
from common import ROOT, GENERATED, DEFAULT_CHECKPOINT, SOURCE, CHECKPOINT_SHA256, SOURCE_REVISION, MODEL_ID, selected_weights, state_name

# Reuse the independent standard-ONNX primitive builder, never the offline
# neural graph, model identity, instance normalization or offline windowing.
offline = ROOT / "tools/model-export"
sys.path.append(str(offline))
spec = importlib.util.spec_from_file_location("reuse_standard_onnx_primitives", offline / "export_model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
BaseGraph = module.Graph


class StreamingGraph(BaseGraph):
    def __init__(self, weights, layers):
        super().__init__(weights)
        self.layers, self.states, self.state_values = layers, [], []

    def cache(self, key, shape):
        suffix = state_name(key)
        state = {"key": key, "inputName": "cache_" + suffix, "outputName": "next_" + suffix,
                 "shape": list(shape), "dtype": "float32"}
        self.states.append(state)
        return state["inputName"]

    def save(self, key, value):
        state = next(s for s in self.states if s["key"] == key)
        self.nodes.append(h.make_node("Identity", [value], [state["outputName"]]))
        self.state_values.append(state)

    def causal(self, x, p, *, channels, frequency, dilation=1, transpose_kernel=False):
        shape = self.weights[p + ".weight"].shape
        kt, kf = (shape[3], shape[2]) if transpose_kernel else (shape[2], shape[3])
        history = (kt - 1) * dilation
        previous = self.cache(p, [1, channels, history, frequency])
        joined = self.cat(previous, x, axis=2)
        self.save(p, self.slice(joined, 1, history + 1, 2))
        weight = self.transpose(p + ".weight", [0, 1, 3, 2]) if transpose_kernel else p + ".weight"
        return self.op("Conv", joined, weight, p + ".bias", pads=[0, kf // 2, 0, kf // 2],
                       dilations=[dilation, 1])

    def norm_prelu(self, x, norm, activation):
        x = self.transpose(x, [0, 2, 3, 1])
        x = self.op("LayerNormalization", x, norm + ".weight", norm + ".bias", axis=-1, epsilon=1e-5)
        x = self.transpose(x, [0, 3, 1, 2])
        return self.op("PRelu", x, self.reshape(activation + ".weight", [1, 64, 1, 1]))

    def dense(self, x, p, frequency):
        skip = x
        for i in range(4):
            q = f"{p}.dense_block.{i}"
            x = self.causal(skip, q + ".1", channels=64 * (i + 1), frequency=frequency, dilation=1 << i)
            x = self.norm_prelu(x, q + ".2.norm", q + ".3")
            if i < 3:
                skip = self.cat(x, skip, axis=1)
        return x

    def decoder(self, x, p):
        x = self.dense(x, p + ".dense_block", 161)
        x = self.op("Pad", x, self.const([0, 0, 0, 1, 0, 0, 0, 1]), mode="constant")
        x = self.conv(x, p + ".up_conv1.0.conv")
        x = self.reshape(x, [1, 2, 64, 1, 161])
        x = self.reshape(self.transpose(x, [0, 2, 3, 4, 1]), [1, 64, 1, 322])
        x = self.norm_prelu(x, p + ".up_conv1.1.norm", p + ".up_conv1.2")
        x = self.causal(x, p + ".up_conv2.0.conv", channels=64, frequency=322, transpose_kernel=True)
        return self.norm_prelu(x, p + ".up_conv2.1.norm", p + ".up_conv2.2")

    def temporal(self, x, p):
        xz = self.linear(x, p + ".in_proj")
        u, z = self.slice(xz, 0, 256, 2), self.slice(xz, 256, 512, 2)
        key = p + ".conv1d"
        old = self.cache(key, [161, 256, 3])
        joined = self.cat(old, self.transpose(u, [0, 2, 1]), axis=2)
        self.save(key, self.slice(joined, 1, 4, 2))
        u = self.op("Conv", joined, p + ".conv1d.weight", p + ".conv1d.bias", group=256)
        u = self.transpose(u, [0, 2, 1])
        u = self.op("Mul", u, self.op("Sigmoid", u))
        projection = self.linear(u, p + ".x_proj", False)
        dt = self.linear(self.slice(projection, 0, 4, 2), p + ".dt_proj", False)
        dt = self.op("Softplus", self.op("Add", dt, p + ".dt_proj.bias"))
        u2 = self.op("Squeeze", u, self.const([1]))
        dt = self.transpose(dt, [0, 2, 1])
        b, c = self.slice(projection, 4, 20, 2), self.slice(projection, 20, 36, 2)
        key = p + ".ssm_state"
        state = self.cache(key, [161, 256, 16])
        a = self.op("Neg", self.op("Exp", p + ".A_log"))
        next_state = self.op("Add", self.op("Mul", self.op("Exp", self.op("Mul", dt, a)), state),
                             self.op("Mul", self.op("Mul", dt, b), self.unsqueeze(u2, [2])))
        self.save(key, next_state)
        y = self.op("ReduceSum", self.op("Mul", next_state, c), self.const([2]), keepdims=0)
        y = self.op("Add", y, self.op("Mul", p + ".D", u2))
        z2 = self.op("Squeeze", z, self.const([1]))
        y = self.op("Mul", y, self.op("Mul", z2, self.op("Sigmoid", z2)))
        return self.linear(self.unsqueeze(y, [1]), p + ".out_proj")

    def build(self):
        x = self.cat(self.unsqueeze(self.transpose("magnitude", [0, 2, 1]), [1]),
                     self.unsqueeze(self.transpose("phase", [0, 2, 1]), [1]), axis=1)
        x = self.op("Pad", x, self.const([0, 0, 0, 0, 0, 0, 0, 2]), mode="constant")
        x = self.causal(x, "dense_encoder.dense_conv_1_1.0", channels=2, frequency=323)
        x = self.norm_prelu(x, "dense_encoder.dense_conv_1_1.1.norm", "dense_encoder.dense_conv_1_1.2")
        x = self.dense(x, "dense_encoder.dense_block", 323)
        x = self.conv(x, "dense_encoder.dense_conv_2.0", stride=(1, 2))
        x = self.norm_prelu(x, "dense_encoder.dense_conv_2.1.norm", "dense_encoder.dense_conv_2.2")
        for i in range(self.layers):
            p = f"TSMamba.{i}"
            tx = self.reshape(self.transpose(x, [0, 3, 2, 1]), [161, 1, 64])
            temporal = self.op("Add", self.temporal(tx, p + ".time_mamba.forward_blocks"), tx)
            temporal = self.linear(temporal, p + ".time_mamba.output_proj")
            temporal = self.op("LayerNormalization", temporal, p + ".time_mamba.norm.weight",
                               p + ".time_mamba.norm.bias", axis=-1, epsilon=1e-5)
            temporal = self.op("Add", temporal, tx)
            fx = self.reshape(temporal, [1, 161, 64])
            fx = self.op("Add", self.bidirectional(fx, p + ".freq_mamba"), fx)
            x = self.transpose(self.reshape(fx, [1, 1, 161, 64]), [0, 3, 1, 2])
        p = f"mask_decoder_list.{self.layers-1}"
        magnitude = self.conv(self.decoder(x, p), p + ".final_conv")
        p = f"phase_decoder_list.{self.layers-1}"
        features = self.decoder(x, p)
        phase = self.atan2(self.conv(features, p + ".phase_conv_i"), self.conv(features, p + ".phase_conv_r"))
        outputs = []
        for value, name in [(magnitude, "enhanced_magnitude"), (phase, "enhanced_phase")]:
            value = self.slice(value, 0, 321, 3)
            value = self.unsqueeze(self.op("Squeeze", value, self.const([1, 2])), [2])
            self.nodes.append(h.make_node("Identity", [value], [name]))
            outputs.append(h.make_tensor_value_info(name, T.FLOAT, [1, 321, 1]))
        inputs = [h.make_tensor_value_info(n, T.FLOAT, [1, 321, 1]) for n in ["magnitude", "phase"]]
        for state in self.states:
            inputs.append(h.make_tensor_value_info(state["inputName"], T.FLOAT, state["shape"]))
            outputs.append(h.make_tensor_value_info(state["outputName"], T.FLOAT, state["shape"]))
        assert len(self.states) == 15 + 2 * self.layers
        graph = h.make_graph(self.nodes, "Stateful one-hop NSCLv1 Real-time RE-USE", inputs, outputs, self.initializers)
        model = h.make_model(graph, producer_name="reuse-audio streaming research exporter",
                             opset_imports=[h.make_opsetid("", 17)])
        model.ir_version = 9
        h.set_model_props(model, {"checkpoint_sha256": CHECKPOINT_SHA256, "source_revision": SOURCE_REVISION,
                                  "precision": "float32", "layers": str(self.layers), "lookahead": "0", "license": "NSCLv1"})
        onnx.checker.check_model(model, full_check=True)
        return model


def export(checkpoint, layers, directory):
    _, weights = selected_weights(checkpoint, layers)
    builder = StreamingGraph(weights, layers)
    model = builder.build()
    path = directory / f"reuse-streaming-l{layers}-fp32.onnx"
    directory.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, path, save_as_external_data=False)
    manifest = {"schemaVersion": 1, "modelId": MODEL_ID, "layers": layers, "lookahead": 0,
                "precision": "float32", "sampleRate": 16000, "fftSize": 640, "hopSize": 320,
                "frameSamples": 640, "frequencyBins": 321, "modelFile": path.name,
                "modelSha256": hashlib.sha256(path.read_bytes()).hexdigest(), "modelBytes": path.stat().st_size,
                "checkpointSha256": CHECKPOINT_SHA256, "checkpointBytes": 54_055_764,
                "sourceRepository": SOURCE["repo"], "sourceRevision": SOURCE_REVISION,
                "sourceParameterCount": 13_475_815, "activeParameterCount": sum(w.numel() for w in weights.values()),
                "license": "NSCLv1", "researchOnly": True, "onnxOpset": 17, "onnxRuntimeVersion": "1.23.2",
                "inputNames": ["magnitude", "phase"] + [s["inputName"] for s in builder.states],
                "outputNames": ["enhanced_magnitude", "enhanced_phase"] + [s["outputName"] for s in builder.states],
                "states": builder.states, "stateBytes": sum(int(np.prod(s["shape"])) * 4 for s in builder.states),
                "frequencyScanCount": layers * 2, "selectedWeightBytesUnchanged": True}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--layers", type=int, nargs="+", default=[4])
    p.add_argument("--output-directory", type=Path, default=GENERATED / "models")
    a = p.parse_args()
    for layer in a.layers:
        info = export(a.checkpoint, layer, a.output_directory)
        print(json.dumps({k: info[k] for k in ["layers", "modelFile", "modelBytes", "modelSha256", "activeParameterCount", "stateBytes"]}), flush=True)


if __name__ == "__main__":
    main()
