#!/usr/bin/env python3
"""Stage pinned research models and synthetic references, never commit weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
BENCHMARK = Path(__file__).resolve().parents[1]


def verify(path, size, sha):
    data = path.read_bytes()
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
        raise ValueError("Pinned asset integrity failed: " + path.name)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "tools/streaming-export/generated/models")
    parser.add_argument("--parity", type=Path, default=ROOT / "tools/streaming-export/generated/parity")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12])
    args = parser.parse_args()
    if not args.layers or not set(args.layers) <= set(range(3, 13)):
        raise ValueError("Choose exported exits 3–12")
    metadata_root = ROOT / "android/reuse-streaming/model"
    catalog_path = metadata_root / "validated-artifacts.json"
    catalog = json.loads(catalog_path.read_text())
    rows = {row["layers"]: row for row in catalog["models"]}
    output = BENCHMARK / "app/build/benchmark-assets"
    output.mkdir(parents=True, exist_ok=True)
    for owned in ("models", "parity", "licenses"):
        path = output / owned
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()
    for depth in sorted(set(args.layers)):
        row = rows[depth]
        for prefix, source in (("model", args.models / row["modelFile"]),
                               ("manifest", metadata_root / "manifests" / row["manifestFile"])):
            data = verify(source, row[prefix + "Bytes"], row[prefix + "Sha256"])
            (output / "models" / row[prefix + "File"]).write_bytes(data)
        receipt_data = verify(args.parity / row["parityFile"], row["parityBytes"], row["paritySha256"])
        receipt = json.loads(receipt_data)
        if receipt["passed"] is not True or receipt["layers"] != depth or receipt["modelSha256"] != row["modelSha256"]:
            raise ValueError("Independent reference identity mismatch")
        destination = output / "parity" / f"l{depth}"
        destination.mkdir()
        (destination / "parity.json").write_bytes(receipt_data)
        for name, record in receipt["files"].items():
            if Path(name).name != name:
                raise ValueError("Unsafe fixture path")
            data = verify(args.parity / f"l{depth}" / name, record["bytes"], record["sha256"])
            (destination / name).write_bytes(data)
    licenses = ROOT / "tools/streaming-export"
    for name in ("LICENSE", "NOTICE", "MAMBA-LICENSE.txt"):
        shutil.copyfile(licenses / name, output / "licenses" / name)
    (output / "benchmark-inputs.json").write_text(json.dumps({
        "sdkCoordinate": "io.github.quangng2000:reuse-streaming-android:0.1.2",
        "layers": sorted(set(args.layers)), "checkpointSha256": catalog["checkpointSha256"],
        "catalogSha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "researchOnly": True, "precision": "float32", "models": [rows[x] for x in sorted(set(args.layers))],
    }, indent=2) + "\n")
    print("Verified/staged FP32 research models and independent synthetic fixtures:", sorted(set(args.layers)))


if __name__ == "__main__":
    main()
