"""Range and edge proof for the final stable library's packet Softplus."""
import ctypes
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import onnxruntime as ort
from onnx import helper as h, TensorProto as T

HERE = Path(__file__).resolve().parent


def main():
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library = HERE / ("generated/build-host/libreuse_streaming_stable_v2" + suffix)
    values = np.concatenate([np.linspace(-120, 120, 1000000, dtype=np.float32),
               np.random.default_rng(234).uniform(-80, 80, 1000000).astype(np.float32),
               np.asarray([-np.inf, np.inf, np.nan, 0., -0., -80., 80., -103.5], np.float32)])
    model = h.make_model(h.make_graph([h.make_node("Softplus", ["x"], ["y"])], "softplus",
                         [h.make_tensor_value_info("x", T.FLOAT, [len(values)])],
                         [h.make_tensor_value_info("y", T.FLOAT, [len(values)])]),
                         opset_imports=[h.make_opsetid("", 17)], ir_version=8)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    expected = session.run(None, {"x": values})[0]
    actual = np.empty_like(values)
    native = ctypes.CDLL(str(library.resolve()))
    function = native.ReuseSoftplusArrayForValidation
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    function(values.ctypes.data, actual.ctypes.data, len(values))
    finite = np.isfinite(expected) & np.isfinite(actual)
    positive = finite & (expected > 0)
    subnormal = (expected > 0) & (expected < np.finfo(np.float32).tiny)
    ulps = abs(expected[finite].view(np.uint32).astype(np.int64) - actual[finite].view(np.uint32).astype(np.int64))
    result = {"count": len(values), "ortVersion": ort.__version__,
              "maxFiniteAbs": float(np.max(abs(expected[finite].astype(np.float64) - actual[finite]))),
              "maxFiniteUlp": int(ulps.max()),
              "maxPositiveRelative": float(np.max(abs(expected[positive].astype(np.float64) - actual[positive]) / expected[positive])),
              "nanClassificationEqual": bool(np.array_equal(np.isnan(expected), np.isnan(actual))),
              "infClassificationEqual": bool(np.array_equal(np.isinf(expected), np.isinf(actual))),
              "positiveClassificationEqual": bool(np.array_equal(expected > 0, actual > 0)),
              "subnormalCount": int(subnormal.sum()), "subnormalBitwiseEqual": bool(np.array_equal(expected[subnormal], actual[subnormal])),
              "fallbackExtremeBitwiseEqual": bool(np.array_equal(expected[abs(values) > 80], actual[abs(values) > 80], equal_nan=True)),
              "nonfiniteFallbackBitwiseEqual": bool(np.array_equal(expected[~np.isfinite(values)], actual[~np.isfinite(values)], equal_nan=True)),
              "nativeLibrarySha256": hashlib.sha256(library.read_bytes()).hexdigest()}
    result["passed"] = (result["maxFiniteAbs"] <= 5e-7 and result["maxFiniteUlp"] <= 4 and result["maxPositiveRelative"] < 1e-5 and
                          all(result[key] for key in ("nanClassificationEqual", "infClassificationEqual", "positiveClassificationEqual",
                               "subnormalBitwiseEqual", "fallbackExtremeBitwiseEqual", "nonfiniteFallbackBitwiseEqual")))
    (HERE / "generated/raw-softplus-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert result["passed"]


if __name__ == "__main__":
    main()
