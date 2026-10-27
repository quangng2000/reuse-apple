"""Prove raw Exp through the final stable library's actual temporal kernel."""
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
    shapes = [[161, 256, 16], [161, 256], [161, 256, 1], [161, 1, 16],
              [161, 1, 16], [161, 256], [256, 16], [256]]
    names = ["old", "u", "dt", "b", "c", "gate", "a", "D"]
    inputs = [h.make_tensor_value_info(name, T.FLOAT, shape) for name, shape in zip(names, shapes)]
    outputs = [h.make_tensor_value_info("next", T.FLOAT, shapes[0]), h.make_tensor_value_info("values", T.FLOAT, shapes[1])]
    node = h.make_node("TemporalStepV2", names, ["next", "values"], domain="ai.reuse.streaming",
                       abi_version=2, channels=256, state_size=16, frequency_bins=161)
    model = h.make_model(h.make_graph([node], "final-library-exp", inputs, outputs),
                         opset_imports=[h.make_opsetid("", 17), h.make_opsetid("ai.reuse.streaming", 1)], ir_version=8)
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    options.register_custom_ops_library(str(library.resolve()))
    native = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    reference = h.make_model(h.make_graph([h.make_node("Mul", ["dt", "a"], ["product"]),
                                           h.make_node("Exp", ["product"], ["exp"])], "reference-exp",
                          [h.make_tensor_value_info("dt", T.FLOAT, shapes[2]), h.make_tensor_value_info("a", T.FLOAT, shapes[6])],
                          [h.make_tensor_value_info("exp", T.FLOAT, shapes[0])]),
                          opset_imports=[h.make_opsetid("", 17)], ir_version=8)
    ref_options = ort.SessionOptions(); ref_options.intra_op_num_threads = 1
    ref = ort.InferenceSession(reference.SerializeToString(), ref_options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(557)
    actuals, expected = [], []
    for index in range(4):
        data = {name: np.zeros(shape, np.float32) for name, shape in zip(names, shapes)}
        data["old"].fill(1)
        data["dt"] = rng.uniform(-70, 50, shapes[2]).astype(np.float32)
        data["a"] = rng.uniform(.5, 2, shapes[6]).astype(np.float32)
        data["a"][0, :8] = [-np.inf, np.inf, np.nan, 0, -0., -103.972084, 88.77626, -88.37626]
        data["dt"][:, 0, 0] = 1.0
        # With old=1, u=0, b=0, next_state = Exp(dt*A)*1 + 0 exactly.
        actuals.append(native.run(["next"], data)[0].reshape(-1))
        expected.append(ref.run(None, {"dt": data["dt"], "a": data["a"]})[0].reshape(-1))
    actual, expected = np.concatenate(actuals), np.concatenate(expected)
    finite = np.isfinite(actual) & np.isfinite(expected)
    ulps = abs(actual[finite].view(np.uint32).astype(np.int64) - expected[finite].view(np.uint32).astype(np.int64))
    result = {"ortVersion": ort.__version__, "count": len(actual), "passed": bool(np.array_equal(actual, expected, equal_nan=True)),
              "finiteBitwiseEqual": bool(np.array_equal(actual[finite], expected[finite])),
              "equalIncludingNanClassification": bool(np.array_equal(actual, expected, equal_nan=True)),
              "maxFiniteUlp": int(ulps.max()), "mismatchedFinite": int(np.count_nonzero(ulps)),
              "infClassificationEqual": bool(np.array_equal(np.isinf(actual), np.isinf(expected))),
              "nanClassificationEqual": bool(np.array_equal(np.isnan(actual), np.isnan(expected))),
              "nativeLibrarySha256": hashlib.sha256(library.read_bytes()).hexdigest(),
              "scope": "Actual final stable TemporalStepV2 raw Exp isolated with old=1,u=0,b=0; four full bounded tensors, random/range/clamp/NaN/Inf cases"}
    (HERE / "generated/raw-exp-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    assert result["passed"]


if __name__ == "__main__":
    main()
