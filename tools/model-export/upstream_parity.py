"""Optional parity against the separately pinned NVIDIA modules + prior CPU adapter.

Reads a caller-provided checkout; never edits it, imports its CLI, downloads
models, or uses GPU. The adapter/source hashes are written into the receipt.
"""
import argparse
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path
import torch
from reference import DEFAULT_CHECKPOINT, Reference, verified_weights
from parity import analyze, error, signal


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-root", required=True, type=Path,
                   help="Directory containing offline-upstream/, mamba_cpu_adapted.py and cpu_scan.py")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--kind", choices=["signal", "silence", "impulse"], default="signal")
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path[:0] = [str(a.reference_root), str(a.reference_root / "offline-upstream")]
    # Optional hub/CUDA libraries are not needed to construct the local modules.
    mamba = importlib.import_module("mamba_cpu_adapted")
    shim = types.ModuleType("mamba_ssm")
    shim.Mamba = mamba.Mamba
    sys.modules["mamba_ssm"] = shim
    hub = types.ModuleType("huggingface_hub")
    hub.PyTorchModelHubMixin = type("PyTorchModelHubMixin", (), {})
    sys.modules["huggingface_hub"] = hub
    from models.generator_SEMamba_time_d4 import SEMamba
    root = a.reference_root / "offline-upstream"
    config = json.loads((root / "config.json").read_text())
    weights = verified_weights(a.checkpoint)
    original = SEMamba(config).eval()
    original.load_state_dict(weights, strict=True)
    mag, pha = analyze(signal(640, a.kind))
    with torch.inference_mode():
        expected_mag, expected_pha, _ = original(mag, pha)
        got_mag, got_pha = Reference(weights)(mag, pha)
    errors = {"magnitude": error(expected_mag.numpy(), got_mag.numpy()),
              "phase": error(expected_pha.numpy(), got_pha.numpy())}
    assert all(x["maxAbs"] < 0.0001 for x in errors.values()), errors
    files = [root / "config.json", root / "models/generator_SEMamba_time_d4.py",
             root / "models/codec_module_time_d4.py", root / "models/mamba_block2_SEMamba.py",
             a.reference_root / "mamba_cpu_adapted.py", a.reference_root / "cpu_scan.py"]
    result = {"passed": True, "samples": 640, "signalKind": a.kind, "numLayers": 30, "strictStateDictLoad": True,
              "sourceHashes": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
              "errors": errors}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
