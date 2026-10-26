"""Pinned NVIDIA Real-time RE-USE identity. NSCLv1: research/education only."""
from pathlib import Path
import hashlib
import json
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
GENERATED = Path(__file__).parent / "generated"
DEFAULT_CHECKPOINT = GENERATED / "realtime-reuse.safetensors"
SOURCE = json.loads((ROOT / "android/reuse-streaming/model/source.json").read_text())
CHECKPOINT_SHA256 = SOURCE["checkpoint"]["sha256"]
SOURCE_REVISION = SOURCE["revision"]
MODEL_ID = "nvidia/realtime-reuse-streaming-fp32"


def selected_weights(path, layers):
    if not 3 <= layers <= 12:
        raise ValueError("Streaming depth must be3..12")
    if hashlib.sha256(path.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Unsupported Real-time RE-USE checkpoint SHA256")
    weights = load_file(str(path), device="cpu")
    if sum(w.numel() for w in weights.values()) != 13_475_815 or any(w.dtype != torch.float32 for w in weights.values()):
        raise ValueError("Expected complete original13475815-parameter FP32 checkpoint")
    def needed(key):
        if key.startswith("dense_encoder.dense_conv_1_"):
            return key.startswith("dense_encoder.dense_conv_1_1.")
        if key.startswith("dense_encoder."):
            return True
        if key.startswith((f"mask_decoder_list.{layers-1}.", f"phase_decoder_list.{layers-1}.")):
            return True
        return key.startswith("TSMamba.") and int(key.split(".")[1]) < layers
    selected = {key: value for key, value in weights.items() if needed(key)}
    if sum(w.numel() for w in selected.values()) != 1_198_019 + 208_512 * layers:
        raise ValueError("Unexpected selected streaming parameter count")
    return weights, selected


def state_name(key):
    return key.replace(".", "_")
