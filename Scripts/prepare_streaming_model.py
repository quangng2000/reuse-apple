"""Fetch the pinned noncommercial streaming checkpoint and its license."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def verify(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path.name}: {actual}")


def fetch(url, path, expected):
    if path.exists():
        verify(path, expected)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".download", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        subprocess.run(["curl", "--fail", "--location", "--proto", "=https",
                        "--proto-redir", "=https", "--retry", "3",
                        "--connect-timeout", "15", "--max-time", "300",
                        url, "--output", str(temporary_path)], check=True)
        verify(temporary_path, expected)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare(destination):
    manifest = json.loads((ROOT / "Research/ModelProvenance.json").read_text())
    checkpoint, license_record = manifest["checkpoint"], manifest["license"]
    base = f"https://huggingface.co/{manifest['repo']}/resolve/{manifest['revision']}"
    license_url = license_record["url"].replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    fetch(license_url, destination.parent / "REALTIME-LICENSE.txt", license_record["sha256"])
    fetch(f"{base}/{checkpoint['upstream_path']}", destination, checkpoint["sha256"])
    print(f"Verified streaming checkpoint: {destination}")
    print("License: NVIDIA NSCLv1 — noncommercial research and education only.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "Resources/realtime-reuse.safetensors",
                        help="Checkpoint destination file (license is saved alongside).")
    prepare(parser.parse_args().output.expanduser().resolve())


if __name__ == "__main__":
    main()
