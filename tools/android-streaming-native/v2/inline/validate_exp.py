"""Bitwise FP32 range/edge comparison against the pinned ORT ARM64 Exp kernel."""
import ctypes
import json
from pathlib import Path
import platform

import numpy as np
from onnx import helper as h, TensorProto as T
import onnxruntime as ort

HERE = Path(__file__).resolve().parent


def main():
    assert ort.__version__ == "1.23.2"
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = HERE / ("generated/build-host/libreuse_streaming_inline_exp_validation" + suffix)
    boundaries = np.asarray([0., -0., -103.972084, 88.77626, -np.inf, np.inf, np.nan, -88.37626], np.float32)
    values = np.concatenate([np.linspace(-140, 100, 1000000, dtype=np.float32),
                             np.random.default_rng(923).uniform(-104, 89, 1000000).astype(np.float32), boundaries])
    model = h.make_model(h.make_graph([h.make_node("Exp", ["x"], ["y"])], "exp",
                          [h.make_tensor_value_info("x", T.FLOAT, [len(values)])],
                          [h.make_tensor_value_info("y", T.FLOAT, [len(values)])]),
                          opset_imports=[h.make_opsetid("", 17)], ir_version=8)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    expected = session.run(None, {"x": values})[0]
    actual = np.empty_like(values)
    native = ctypes.CDLL(str(library.resolve()))
    compute = native.ReuseExpArrayForValidation
    compute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    compute(values.ctypes.data, actual.ctypes.data, len(values))
    finite = np.isfinite(expected) & np.isfinite(actual)
    ulps = abs(expected[finite].view(np.uint32).astype(np.int64) - actual[finite].view(np.uint32).astype(np.int64))
    result = {"ortVersion": ort.__version__, "count": len(values),
              "finiteBitwiseEqual": bool(np.array_equal(expected[finite], actual[finite])),
              "equalIncludingNanClassification": bool(np.array_equal(expected, actual, equal_nan=True)),
              "maxFiniteUlp": int(ulps.max()), "mismatchedFinite": int(np.count_nonzero(ulps)),
              "infClassificationEqual": bool(np.array_equal(np.isinf(expected), np.isinf(actual))),
              "nanClassificationEqual": bool(np.array_equal(np.isnan(expected), np.isnan(actual))),
              "scope": "Two million FP32 range/random inputs plus zero, underflow/overflow clamps, infinities and NaN"}
    (HERE / "generated/raw-exp-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert result["equalIncludingNanClassification"]


if __name__ == "__main__":
    main()
