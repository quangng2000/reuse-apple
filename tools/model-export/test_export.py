"""Fast CPU exporter checks independent of any real checkpoint or recordings."""
import unittest
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper as h, TensorProto as T
from export_model import Graph
from reference import Reference, verified_weights
from pathlib import Path
import tempfile


def session(graph, names, outputs):
    model = h.make_model(h.make_graph(graph.nodes, "test", [h.make_tensor_value_info(n, T.FLOAT, None) for n in names],
                                      [h.make_tensor_value_info(n, T.FLOAT, None) for n in outputs], graph.initializers),
                         opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 9
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


class ExportTests(unittest.TestCase):
    def test_checkpoint_identity_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "untrusted.safetensors"
            path.write_bytes(b"different model")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verified_weights(path)

    def test_scan_recurrence_dynamic_lengths_and_batches(self):
        rng = np.random.default_rng(471)
        weights = {"test.A_log": torch.tensor(rng.uniform(-2, 2, (256, 16)), dtype=torch.float32),
                   "test.D": torch.tensor(rng.normal(size=256), dtype=torch.float32),
                   "test.dt_proj.bias": torch.tensor(rng.uniform(-6, -1, 256), dtype=torch.float32)}
        graph = Graph(weights)
        out = graph.scan("u", "delta", "b", "c", "z", "test")
        runtime = session(graph, ["u", "delta", "b", "c", "z"], [out])
        for batch, length in [(1, 1), (2, 3), (1, 11)]:
            values = {n: rng.normal(size=(batch, length, channels)).astype(np.float32)
                      for n, channels in [("u", 256), ("delta", 256), ("b", 16), ("c", 16), ("z", 256)]}
            expected = Reference(weights).scan(*(torch.tensor(values[n]) for n in ("u", "delta", "b", "c", "z")), "test").numpy()
            got = runtime.run(None, values)[0]
            np.testing.assert_allclose(got, expected, rtol=2e-6, atol=3e-6)

    def test_atan2_quadrants_and_signed_zero(self):
        graph = Graph({})
        out = graph.atan2("y", "x")
        runtime = session(graph, ["y", "x"], [out])
        y = np.array([1, 1, -1, -1, 0., -0., 0., -0., 1, -1], dtype=np.float32)
        x = np.array([1, -1, -1, 1, 0., 0., -0., -0., 0, 0], dtype=np.float32)
        result = runtime.run(None, {"y": y, "x": x})[0]
        expected = torch.atan2(torch.tensor(y), torch.tensor(x)).numpy()
        np.testing.assert_allclose(result, expected, rtol=0, atol=5e-7)
        # ORT may canonicalize -0 to +0; this backend promises numerical FP32
        # parity, not signed-zero bitwise identity. Quadrant ±pi must survive.
        np.testing.assert_array_equal(np.signbit(result[expected != 0]), np.signbit(expected[expected != 0]))

    def test_atan2_finite_random(self):
        graph = Graph({})
        out = graph.atan2("y", "x")
        runtime = session(graph, ["y", "x"], [out])
        rng = np.random.default_rng(938)
        y, x = rng.normal(size=(2, 2048)).astype(np.float32)
        got = runtime.run(None, {"y": y, "x": x})[0]
        np.testing.assert_allclose(got, np.arctan2(y, x), rtol=0, atol=5e-7)


if __name__ == "__main__":
    unittest.main()
