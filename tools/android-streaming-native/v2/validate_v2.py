"""Validate the opt-in native graph against released ORT and the independent CPU oracle.

No audio alignment, gain correction, state substitution, or precision conversion is
applied. This is synthetic numerical validation, not a broad listening-quality test.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools/streaming-export"))
from common import DEFAULT_CHECKPOINT
from oracle import Oracle
from parity import analyze_audio, synthesize_audio
sys.path.insert(0, str(HERE.parent))
from transform import SOURCE_SHA, DOMAIN, digest


def errors(expected, actual):
    assert expected.shape == actual.shape
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    diff = expected.astype(np.float64) - actual.astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    return {"maxAbs": float(abs(diff).max()), "rmse": rmse,
            "relativeRms": rmse / rms if rms else (0. if rmse == 0 else None),
            "bitwiseEqual": bool(np.array_equal(expected, actual))}


def session(model, library, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if library:
        options.register_custom_ops_library(str(library.resolve()))
    # Retain the options alongside its session because it owns custom-op resources.
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"]), options


class Runner:
    def __init__(self, engine, manifest):
        self.engine, self.info = engine, manifest
        self.reset()

    def reset(self):
        self.state = {spec["inputName"]: np.zeros(spec["shape"], np.float32) for spec in self.info["states"]}

    def step(self, mag, phase):
        values = self.engine.run(self.info["outputNames"], dict(magnitude=mag, phase=phase, **self.state))
        for spec, value in zip(self.info["states"], values[2:]):
            assert value.dtype == np.float32 and value.shape == tuple(spec["shape"])
            assert np.isfinite(value).all()
            self.state[spec["inputName"]] = value
        assert all(value.dtype == np.float32 and np.isfinite(value).all() for value in values[:2])
        return values[0], values[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    parser.add_argument("--library", type=Path, default=HERE / ("generated/build-host/libreuse_streaming_native_v2" + suffix))
    parser.add_argument("--model", type=Path, default=HERE / "generated/reuse-streaming-l8-native-frequency-temporal-v2.onnx")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=HERE / "generated/validation.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert ort.__version__ == "1.23.2"
    baseline = ROOT / "tools/streaming-export/generated/models/reuse-streaming-l8-fp32.onnx"
    assert digest(baseline) == SOURCE_SHA
    info = json.loads(args.model.with_suffix(".manifest.json").read_text())
    assert digest(args.model) == info["modelSha256"]
    assert info["baselineModelSha256"] == SOURCE_SHA
    assert info["layers"] == 8 and info["precision"] == "float32" and info["lookahead"] == 0
    original, transformed = onnx.load(baseline), onnx.load(args.model)
    before = {item.name: item.SerializeToString() for item in original.graph.initializer}
    after = {item.name: item.SerializeToString() for item in transformed.graph.initializer}
    assert all(after[name] == value for name, value in before.items())
    assert [item.SerializeToString() for item in original.graph.input] == [item.SerializeToString() for item in transformed.graph.input]
    assert [item.SerializeToString() for item in original.graph.output] == [item.SerializeToString() for item in transformed.graph.output]
    assert sum(node.op_type == "Scan" for node in transformed.graph.node) == 0
    assert sum(node.domain == DOMAIN and node.op_type == "FrequencyScanV1" for node in transformed.graph.node) == 16
    assert sum(node.domain == DOMAIN and node.op_type == "TemporalStepV2" for node in transformed.graph.node) == 8
    standard, standard_options = session(baseline, None, args.threads)
    custom, custom_options = session(args.model, args.library, args.threads)
    a, b = Runner(standard, info), Runner(custom, info)
    oracle = Oracle(args.checkpoint, 8)
    started = time.monotonic()
    rng = np.random.default_rng(241024)
    magnitudes = rng.uniform(0, 1, (8, 1, 321, 1)).astype(np.float32)
    phases = rng.uniform(-np.pi, np.pi, magnitudes.shape).astype(np.float32)
    max_state_abs, state_bitwise, state_count, state_scaled = 0., 0, 0, 0.
    expected_mag, actual_mag, expected_phase, actual_phase = [], [], [], []
    oracle_mag, oracle_phase, first, final = [], [], None, None
    for mag, phase in zip(magnitudes, phases):
        am, ap = a.step(mag, phase)
        bm, bp = b.step(mag, phase)
        om, op = oracle.step(mag, phase)
        if first is None:
            first = (bm.copy(), bp.copy(), {name: value.copy() for name, value in b.state.items()})
        expected_mag.append(am); actual_mag.append(bm); expected_phase.append(ap); actual_phase.append(bp)
        oracle_mag.append(om); oracle_phase.append(op)
        oracle_states = oracle.states()
        for spec in info["states"]:
            av, bv = a.state[spec["inputName"]], b.state[spec["inputName"]]
            result = errors(av, bv)
            max_state_abs = max(max_state_abs, result["maxAbs"])
            state_bitwise += result["bitwiseEqual"]; state_count += 1
            state_scaled = max(state_scaled, float(np.max(abs(av.astype(np.float64) - bv) / (.0003 + .0002 * abs(av)))))
            assert np.allclose(av, bv, atol=.0003, rtol=.0002), (spec["key"], result)
            assert np.allclose(oracle_states[spec["key"]], bv, atol=.0003, rtol=.0002), spec["key"]
        final = bm.copy()
    b.reset()
    reset_m, reset_p = b.step(magnitudes[0], phases[0])
    assert np.array_equal(reset_m, first[0]) and np.array_equal(reset_p, first[1])
    assert all(np.array_equal(b.state[name], value) for name, value in first[2].items())
    b.reset()
    fresh_m, _ = b.step(magnitudes[-1], phases[-1])
    carry = errors(final, fresh_m)["maxAbs"]
    assert carry > 1e-6
    expected_mag, actual_mag = np.stack(expected_mag), np.stack(actual_mag)
    expected_phase, actual_phase = np.stack(expected_phase), np.stack(actual_phase)
    wrapped = np.angle(np.exp(1j * (expected_phase.astype(np.float64) - actual_phase)))
    oracle_wrapped = np.angle(np.exp(1j * (np.stack(oracle_phase).astype(np.float64) - actual_phase)))
    assert errors(expected_mag, actual_mag)["maxAbs"] < .003
    assert float(abs(wrapped).max()) < .003
    assert float(abs(oracle_wrapped).max()) < .003
    receipts = []
    for count in (640, 641, 959, 960, 1377, 8000, 16000):
        t = np.arange(count, dtype=np.float64) / 16000
        samples = (.17*np.sin(2*np.pi*173*t) + .08*np.sin(2*np.pi*917*t) + .013*np.random.default_rng(421).standard_normal(count)).astype(np.float32)
        im, ip = analyze_audio(samples)
        a.reset(); b.reset(); oracle.reset()
        outputs = [([], []), ([], []), ([], [])]
        for mag, phase in zip(im, ip):
            for index, pair in enumerate((a.step(mag, phase), b.step(mag, phase), oracle.step(mag, phase))):
                outputs[index][0].append(pair[0]); outputs[index][1].append(pair[1])
        waves = [synthesize_audio(*pair, count) for pair in outputs]
        baseline_error, oracle_error = errors(waves[0], waves[1]), errors(waves[2], waves[1])
        assert baseline_error["maxAbs"] < .0001 and baseline_error["relativeRms"] < .001
        assert oracle_error["maxAbs"] < .0003
        receipts.append({"samples": count, "nativeVsReleased": baseline_error, "nativeVsIndependentOracle": oracle_error})
    receipt = {"passed": True, "layers": 8, "precision": "float32", "lookahead": 0,
               "ortVersion": ort.__version__, "threads": args.threads, "baselineSha256": SOURCE_SHA,
               "nativeModelSha256": digest(args.model), "nativeLibrarySha256": digest(args.library),
               "allOriginalInitializerBytesUnchanged": True, "allExternalInputOutputSchemasUnchanged": True,
               "nativeFrequencyScanCount": 16, "nativeTemporalStepCount": 8, "persistentStateCount": len(info["states"]),
               "spectralHops": 8, "magnitudeError": errors(expected_mag, actual_mag),
               "wrappedPhaseError": errors(np.zeros_like(wrapped), wrapped),
               "nativeVsIndependentMagnitude": errors(np.stack(oracle_mag), actual_mag),
               "nativeVsIndependentWrappedPhase": errors(np.zeros_like(oracle_wrapped), oracle_wrapped),
               "maxStateAbs": max_state_abs, "maxNormalizedStateTolerance": state_scaled,
               "stateAtol": .0003, "stateRtol": .0002,
               "bitwiseStateOutputs": state_bitwise, "stateOutputs": state_count,
               "resetAllOutputsAndStatesBitwiseEqual": True, "carriedStateOutputDifferenceFromFresh": carry,
               "waveformMaxAbsThresholdVsReleased": .0001, "waveformRelativeRmsThresholdVsReleased": .001,
               "waveformMaxAbsThresholdVsIndependentOracle": .0003, "audioFixtures": receipts,
               "elapsedSeconds": time.monotonic() - started,
               "scope": "Unaligned synthetic waveform and all carried-state numerical parity; not listening validation or physical-device speed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
