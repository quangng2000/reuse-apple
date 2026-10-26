"""Run all-layer PyTorch/CPU-ORT parity and generate portable raw FP32 fixtures.

No audio recordings are read: every fixture is a deterministic synthetic signal.
Numerical checks are explicit FP32 error checks, never a bitwise-equivalence claim.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from reference import CHECKPOINT_SHA256, Reference, verified_weights


def error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(f"Invalid output: {a.shape}, {b.shape}")
    d = a.astype(np.float64) - b.astype(np.float64)
    return {"shape": list(a.shape), "maxAbs": float(np.max(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d * d))),
            "referencePeak": float(np.max(np.abs(a))),
            "referenceRMS": float(np.sqrt(np.mean(a.astype(np.float64) ** 2))),
            "actualPeak": float(np.max(np.abs(b))),
            "actualRMS": float(np.sqrt(np.mean(b.astype(np.float64) ** 2))),
            "bitwiseEqual": bool(np.array_equal(a, b))}


def signal(count, kind="signal"):
    if kind == "silence":
        return np.zeros(count, dtype=np.float32)
    if kind == "impulse":
        x = np.zeros(count, dtype=np.float32)
        x[count // 2] = .5
        return x
    t = np.arange(count, dtype=np.float64) / 16000
    rng = np.random.default_rng(421)
    return (0.17 * np.sin(2 * np.pi * 173 * t) + 0.08 * np.sin(2 * np.pi * 917 * t)
            + 0.013 * rng.standard_normal(count)).astype(np.float32)


def si_sdr(clean, estimate):
    clean = clean.astype(np.float64) - np.mean(clean, dtype=np.float64)
    estimate = estimate.astype(np.float64) - np.mean(estimate, dtype=np.float64)
    target = clean * (np.dot(clean, estimate) / max(np.dot(clean, clean), 1e-20))
    return float(10 * np.log10(max(np.dot(target, target), 1e-20) /
                              max(np.dot(estimate - target, estimate - target), 1e-20)))


def analyze(x):
    # Match the existing Apple wrapper: min640 and round up to hop80, then
    # centered reflect padding, periodic Hann, unnormalized real FFT.
    n = max(640, ((len(x) + 79) // 80) * 80)
    x = torch.nn.functional.pad(torch.tensor(x), (0, n - len(x)))
    spec = torch.stft(x, 640, 80, 640, torch.hann_window(640), center=True,
                      pad_mode="reflect", normalized=False, return_complex=True)
    return torch.log1p(spec.abs())[None], torch.angle(spec)[None]


def synthesize(magnitude, phase, count):
    mag = torch.expm1(torch.clamp_min(torch.tensor(magnitude[0]), 0))
    empty = (mag == 0).float().mean(dim=0) > .5
    mag[:, empty] = 0
    spec = torch.complex(mag * torch.cos(torch.tensor(phase[0])), mag * torch.sin(torch.tensor(phase[0])))
    return torch.istft(spec, 640, 80, 640, torch.hann_window(640), center=True, length=count).numpy()


def write_fixture(directory, name, x):
    p = directory / (name + ".f32")
    np.asarray(x, dtype="<f4").tofile(p)
    return {"file": p.name, "shape": list(x.shape), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--samples", nargs="+", type=int, default=[640, 8000, 64000])
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "generated/parity")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--kind", choices=["signal", "silence", "impulse"], default="signal")
    p.add_argument("--input-wav", type=Path, help="Explicit paired speech fixture, mono16k WAV only")
    p.add_argument("--clean-wav", type=Path, help="Optional matching clean fixture for STOI")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    weights = verified_weights(a.checkpoint)
    ref = Reference(weights)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = a.threads, 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # No reduced-precision EP, NNAPI, CoreML or CUDA is registered.
    session = ort.InferenceSession(str(a.model), options, providers=["CPUExecutionProvider"])
    names = [x.name for x in session.get_outputs()]
    graph = onnx.load(a.model)
    scan_nodes = [n for n in graph.graph.node if n.op_type == "Scan"]
    assert len(scan_nodes) == 120
    assert all(not n.domain for n in graph.graph.node)
    assert all(x.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.INT64) for x in graph.graph.initializer)
    initializers = {x.name: onnx.numpy_helper.to_array(x) for x in graph.graph.initializer}
    assert all(k in initializers and initializers[k].dtype == np.float32
               and initializers[k].shape == tuple(w.shape)
               and initializers[k].tobytes() == w.numpy().tobytes() for k, w in weights.items())
    receipt = {"checkpointSha256": CHECKPOINT_SHA256,
               "modelSha256": hashlib.sha256(a.model.read_bytes()).hexdigest(),
               "torchVersion": torch.__version__, "onnxVersion": onnx.__version__,
               "onnxRuntimeVersion": ort.__version__, "providers": session.get_providers(),
               "scanCount": len(scan_nodes), "numLayers": 30,
               "originalWeightBytesUnchanged": True, "learnedTensorCount": len(weights),
               "parameterCount": sum(x.numel() for x in weights.values()), "signalKind": a.kind,
               "scope": "all-layer synthetic FP32 functional parity; not a perceptual-quality or Android-device benchmark",
               "tolerance": {"diagnosticSpectralMaxAbs": .003, "waveformMaxAbs": .0003,
                             "pairedSTOIAbsDelta": .0001, "pairedSISDRAbsDeltaDB": .02}, "cases": []}
    def read_wav(path):
        from scipy.io import wavfile
        rate, data = wavfile.read(path)
        if rate != 16000 or data.ndim != 1:
            raise ValueError("Paired fixture must be mono16kHz")
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768
        elif data.dtype not in (np.float32, np.float64):
            raise ValueError("Supported fixture encodings are PCM16 or float32/64")
        return np.asarray(data, dtype=np.float32)

    source = read_wav(a.input_wav) if a.input_wav else None
    clean = read_wav(a.clean_wav) if a.clean_wav else None
    if clean is not None and (source is None or source.shape != clean.shape):
        raise ValueError("Clean and input fixture lengths must match")
    if source is not None:
        receipt["sourceInputSha256"] = hashlib.sha256(a.input_wav.read_bytes()).hexdigest()
        receipt["signalKind"] = "explicit-speech-fixture"
    if clean is not None:
        receipt["sourceCleanSha256"] = hashlib.sha256(a.clean_wav.read_bytes()).hexdigest()
    for samples in ([len(source)] if source is not None else a.samples):
        x = source if source is not None else signal(samples, a.kind)
        mag, pha = analyze(x)
        intermediate = {}
        start = time.monotonic()
        pm, pp = ref(mag, pha, inspect=lambda n, v: intermediate.__setitem__(n, v.numpy().copy()))
        reference_seconds = time.monotonic() - start
        start = time.monotonic()
        actual = dict(zip(names, session.run(None, {"magnitude": mag.numpy(), "phase": pha.numpy()})))
        ort_seconds = time.monotonic() - start
        errors = {"magnitude": error(pm.numpy(), actual["enhanced_magnitude"])}
        # Phase is circular: report wrapped error to avoid the equivalent ±pi boundary.
        phase_delta = np.angle(np.exp(1j * (pp.numpy().astype(np.float64) - actual["enhanced_phase"])))
        errors["phaseWrapped"] = error(np.zeros_like(phase_delta), phase_delta)
        complex_ref = np.expm1(np.maximum(pm.numpy(), 0)) * np.exp(1j * pp.numpy())
        complex_got = np.expm1(np.maximum(actual["enhanced_magnitude"], 0)) * np.exp(1j * actual["enhanced_phase"])
        cd = np.abs(complex_ref - complex_got).astype(np.float64)
        errors["complexSpectrum"] = {"maxAbs": float(cd.max()), "rmse": float(np.sqrt(np.mean(cd * cd))),
                                     "referenceRMS": float(np.sqrt(np.mean(np.abs(complex_ref).astype(np.float64) ** 2)))}
        for n, v in intermediate.items():
            if n in actual:
                errors[n] = error(v, actual[n])
        padded = max(640, ((samples + 79) // 80) * 80)
        wave_ref = synthesize(pm.numpy(), pp.numpy(), padded)
        wave_ort = synthesize(actual["enhanced_magnitude"], actual["enhanced_phase"], padded)
        errors["waveform"] = error(wave_ref, wave_ort)
        case = {"samples": samples, "frames": mag.shape[-1], "referenceSeconds": reference_seconds,
                "ortSeconds": ort_seconds, "errors": errors,
                "fixtures": {n: write_fixture(a.output, f"{samples}-{n}", v) for n, v in {
                    "input": x, "magnitude": mag.numpy(), "phase": pha.numpy(),
                    "reference_magnitude": pm.numpy(), "reference_phase": pp.numpy(),
                    "reference_waveform": wave_ref[:samples],
                    "enhanced_magnitude": actual["enhanced_magnitude"],
                    "enhanced_phase": actual["enhanced_phase"], "waveform": wave_ort[:samples]}.items()}}
        if clean is not None:
            from pystoi import stoi
            case["quality"] = {"metric": "pystoi.stoi(...,extended=False), one paired synthetic-crackle diagnostic only",
                               "inputSTOI": float(stoi(clean, x, 16000, extended=False)),
                               "referenceSTOI": float(stoi(clean, wave_ref[:samples], 16000, extended=False)),
                               "ortSTOI": float(stoi(clean, wave_ort[:samples], 16000, extended=False)),
                               "inputSISDRDB": si_sdr(clean, x),
                               "referenceSISDRDB": si_sdr(clean, wave_ref[:samples]),
                               "ortSISDRDB": si_sdr(clean, wave_ort[:samples])}
            q = case["quality"]
            case["pairedQualityPassed"] = (abs(q["referenceSTOI"] - q["ortSTOI"]) <= .0001 and
                                            abs(q["referenceSISDRDB"] - q["ortSISDRDB"]) <= .02)
        case["waveformPassed"] = errors["waveform"]["maxAbs"] <= .0003
        case["rawPhasePassed"] = errors["phaseWrapped"]["maxAbs"] <= .003
        receipt["cases"].append(case)
        # Raw phase is ill-conditioned for nearly-real/near-zero spectral bins.
        # Keep it visible, including failures; audio acceptance remains the
        # original fixed waveform bound plus the paired speech quality gates.
        case["spectralDiagnosticsPassed"] = errors["magnitude"]["maxAbs"] <= .003 and case["rawPhasePassed"]
        passed = case["waveformPassed"] and case.get("pairedQualityPassed", True)
        case["passed"] = passed
        (a.output / "parity.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"samples": samples, "passed": passed, "referenceSeconds": reference_seconds,
                          "ortSeconds": ort_seconds, "errors": errors}), flush=True)
        if not passed:
            raise AssertionError("Fixed waveform/paired-quality thresholds failed; do not loosen them without investigation")


if __name__ == "__main__":
    main()
