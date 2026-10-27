"""Focused structural, provenance, and fail-closed native loader tests."""
import json
from pathlib import Path
import platform
import unittest

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transform import DOMAIN, SOURCE_SHA, digest

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parents[2] / "tools/streaming-export/generated/models/reuse-streaming-l8-fp32.onnx"
MODEL = HERE / "generated/reuse-streaming-l8-native-frequency-temporal-v2.onnx"
LIBRARY = HERE / ("generated/build-host/libreuse_streaming_native_v2" + (".dylib" if platform.system() == "Darwin" else ".so"))


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = onnx.load(BASELINE)
        cls.model = onnx.load(MODEL)

    def native_session(self, model):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.register_custom_ops_library(str(LIBRARY.resolve()))
        return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"]), options

    def test_baseline_and_artifact_hashes(self):
        self.assertEqual(digest(BASELINE), SOURCE_SHA)
        info = json.loads(MODEL.with_suffix(".manifest.json").read_text())
        self.assertEqual(digest(MODEL), info["modelSha256"])
        self.assertEqual(SOURCE_SHA, info["baselineModelSha256"])
        self.assertEqual(8, info["layers"])
        self.assertEqual("float32", info["precision"])
        self.assertEqual(0, info["lookahead"])
        from build_v2 import source_receipt
        self.assertEqual(source_receipt(), info["nativeSources"])

    def test_header_provenance(self):
        directory = HERE.parent / "vendor/onnxruntime-1.23.2"
        for entry in json.loads((directory / "provenance.json").read_text())["headers"]:
            self.assertEqual(digest(directory / entry["file"]), entry["sha256"])
            self.assertEqual((directory / entry["file"]).stat().st_size, entry["bytes"])

    def test_all_original_initializers_and_external_states_unchanged(self):
        before = {item.name: item.SerializeToString() for item in self.original.graph.initializer}
        after = {item.name: item.SerializeToString() for item in self.model.graph.initializer}
        self.assertTrue(all(after[name] == value for name, value in before.items()))
        for kind in ("input", "output"):
            self.assertEqual([item.SerializeToString() for item in getattr(self.original.graph, kind)],
                             [item.SerializeToString() for item in getattr(self.model.graph, kind)])
        self.assertEqual(16, sum(node.op_type == "Scan" for node in self.original.graph.node))
        self.assertEqual(0, sum(node.op_type == "Scan" for node in self.model.graph.node))
        self.assertEqual(16, sum(node.domain == DOMAIN and node.op_type == "FrequencyScanV1" for node in self.model.graph.node))
        self.assertEqual(8, sum(node.domain == DOMAIN and node.op_type == "TemporalStepV2" for node in self.model.graph.node))

    def test_unregistered_model_refuses_to_load(self):
        options = ort.SessionOptions(); options.intra_op_num_threads = 1
        with self.assertRaisesRegex(Exception, "not a registered function/op"):
            ort.InferenceSession(str(MODEL), options, providers=["CPUExecutionProvider"])

    def test_native_model_loads(self):
        session, options = self.native_session(str(MODEL))
        self.assertEqual(33, len(session.get_inputs()))
        self.assertEqual(33, len(session.get_outputs()))

    def check_attribute_refused(self, name, value, pattern):
        model = onnx.ModelProto(); model.CopyFrom(self.model)
        node = next(item for item in model.graph.node if item.domain == DOMAIN and item.op_type == "FrequencyScanV1")
        attribute = next(item for item in node.attribute if item.name == name)
        attribute.i = value
        with self.assertRaisesRegex(Exception, pattern):
            self.native_session(model.SerializeToString())

    def test_bad_abi_refused(self):
        self.check_attribute_refused("abi_version", 2, "Unsupported native frequency ABI")

    def test_bad_topology_refused(self):
        self.check_attribute_refused("frequency_bins", 160, "Unsupported native frequency topology")

    def test_bad_parallel_mode_refused(self):
        self.check_attribute_refused("parallel", 2, "Unsupported native frequency parallel mode")

    def test_temporal_wrong_abi_refused(self):
        model = onnx.ModelProto(); model.CopyFrom(self.model)
        node = next(item for item in model.graph.node if item.op_type == "TemporalStepV2")
        next(item for item in node.attribute if item.name == "abi_version").i = 1
        with self.assertRaisesRegex(Exception, "Unsupported native temporal ABI"):
            self.native_session(model.SerializeToString())

    def test_library_declares_abi_two(self):
        import ctypes
        library = ctypes.CDLL(str(LIBRARY.resolve()))
        library.ReuseStreamingKernelAbiVersion.restype = ctypes.c_int
        self.assertEqual(2, library.ReuseStreamingKernelAbiVersion())

    def test_temporal_exact_fp32_mul_add_and_reduction_order(self):
        shapes = [[161, 256, 16], [161, 256], [161, 256, 1], [161, 1, 16],
                  [161, 1, 16], [161, 256], [161, 256, 16], [256]]
        names = ["old", "u", "dt", "b", "c", "gate", "decay", "D"]
        node = helper.make_node("TemporalStepV2", names, ["next", "values"], domain=DOMAIN,
                                abi_version=2, channels=256, state_size=16, frequency_bins=161)
        inputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape) for name, shape in zip(names, shapes)]
        outputs = [helper.make_tensor_value_info("next", onnx.TensorProto.FLOAT, shapes[0]),
                   helper.make_tensor_value_info("values", onnx.TensorProto.FLOAT, shapes[1])]
        model = helper.make_model(helper.make_graph([node], "temporal-arithmetic", inputs, outputs),
                                   opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(DOMAIN, 1)], ir_version=8)
        session, options = self.native_session(model.SerializeToString())
        rng = np.random.default_rng(323)
        values = {name: rng.uniform(-.5, .5, shape).astype(np.float32) for name, shape in zip(names, shapes)}
        expected_state = values["decay"] * values["old"] + (values["dt"] * values["b"]) * values["u"][:, :, None]
        product = expected_state * values["c"]
        summed = np.zeros(shapes[1], np.float32)
        for index in range(16):
            summed = summed + product[:, :, index]
        expected_values = (summed + values["D"] * values["u"]) * values["gate"]
        actual_state, actual_values = session.run(None, values)
        self.assertTrue(np.array_equal(expected_state, actual_state))
        self.assertTrue(np.array_equal(expected_values, actual_values))

    def test_input_shape_rejected_in_kernel(self):
        shapes = [[1, 256, 16], [1, 161, 256], [1, 161, 256], [1, 161, 16],
                  [1, 161, 16], [1, 161, 256], [1, 161, 256, 16], [256]]
        names = ["initial", "u", "dt", "b", "c", "gate", "decay", "D"]
        node = helper.make_node("FrequencyScanV1", names, ["final", "values"], domain=DOMAIN,
                                abi_version=1, channels=256, state_size=16, frequency_bins=161, parallel=0)
        inputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT,
                  [None] * len(shape)) for name, shape in zip(names, shapes)]
        outputs = [helper.make_tensor_value_info("final", onnx.TensorProto.FLOAT, shapes[0]),
                   helper.make_tensor_value_info("values", onnx.TensorProto.FLOAT, shapes[1])]
        model = helper.make_model(helper.make_graph([node], "negative-shape", inputs, outputs),
                                   opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(DOMAIN, 1)], ir_version=8)
        session, options = self.native_session(model.SerializeToString())
        values = {name: np.zeros(shape, np.float32) for name, shape in zip(names, shapes)}
        values["u"] = np.zeros([1, 160, 256], np.float32)
        with self.assertRaisesRegex(Exception, "exact bounded FP32 shapes"):
            session.run(None, values)


if __name__ == "__main__":
    unittest.main()
