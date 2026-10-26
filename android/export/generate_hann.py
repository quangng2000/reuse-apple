"""Verify/reproduce the sealed640-sample FP32 periodic Hann coefficients."""
import argparse
import hashlib
from pathlib import Path
import torch

SHA256 = "ca961f55e3d74499159074e0401c180a88f41c6a4a7d9540fb733cb3d8a91a27"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "generated/hann640.f32")
    a = p.parse_args()
    if torch.__version__.split("+")[0] != "2.10.0":
        raise RuntimeError("Reproduction requires pinned PyTorch2.10.0; verify the output hash on each platform")
    values = torch.hann_window(640, periodic=True, dtype=torch.float32).numpy()
    data = values.astype("<f4").tobytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != SHA256:
        raise RuntimeError(f"Different Hann rounding on this platform: {actual}; do not replace the sealed coefficients")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(data)
    print(f"{a.output}: 640 FP32 coefficients, SHA256 {actual}")


if __name__ == "__main__":
    main()
