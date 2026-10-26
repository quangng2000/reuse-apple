#!/usr/bin/env python3
"""Fetch immutable, hash-pinned research models and synthetic release fixtures.

Trust comes from the checked-in export catalog, never a downloaded manifest.
No publisher credentials are used. Model weights stay outside source/Maven.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "android/reuse-streaming/model/validated-artifacts.json"
RELEASE_TAG = "android-streaming-models-v1"
RELEASE_URL = "https://github.com/quangng2000/reuse-audio/releases/download/" + RELEASE_TAG + "/"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_bytes(data, record, label):
    require(isinstance(record["bytes"], int) and record["bytes"] > 0, "Invalid pinned byte count")
    require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None, "Invalid pinned SHA-256")
    require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"],
            label + " failed size/SHA-256 verification")


def asset_record(row, prefix):
    return {"file": row[prefix + "File"], "bytes": row[prefix + "Bytes"],
            "sha256": row[prefix + "Sha256"]}


def download(record, directory):
    name = record["file"]
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", name) is not None, "Invalid release asset name")
    require(0 < record["bytes"] <= 32 * 1024 * 1024, "Release asset exceeds size limit")
    request = urllib.request.Request(RELEASE_URL + name, headers={"User-Agent": "reuse-audio-streaming-ci"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read(record["bytes"] + 1)
    checked_bytes(data, record, name)
    destination = directory / name
    destination.write_bytes(data)
    return destination


def extract_parity(archive_path, destination, catalog):
    """Validate every archive entry before writing any fixture."""
    rows = catalog["models"]
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        require(len(names) == len(set(names)), "Duplicate parity archive paths")
        require(len(entries) <= 256 and sum(e.file_size for e in entries) <= 64 * 1024 * 1024,
                "Parity archive expansion limit")
        for entry in entries:
            path = PurePosixPath(entry.filename)
            require(not entry.is_dir() and not path.is_absolute() and ".." not in path.parts and
                    "\\" not in entry.filename and entry.file_size <= 2 * 1024 * 1024 and
                    ((entry.external_attr >> 16) & 0o170000) != 0o120000,
                    "Unsafe parity archive entry")
        expected = {}
        for row in rows:
            name = row["parityFile"]
            require(name == f'l{row["layers"]}/parity.json', "Unexpected parity receipt path")
            expected[name] = asset_record(row, "parity")
            require(name in names, "Missing parity receipt")
            data = archive.read(name)
            checked_bytes(data, expected[name], name)
            receipt = json.loads(data)
            require(receipt["schemaVersion"] == 1 and receipt["passed"] is True and
                    receipt["layers"] == row["layers"] and receipt["modelSha256"] == row["modelSha256"] and
                    receipt["checkpointSha256"] == catalog["checkpointSha256"] and receipt["dtype"] == "float32",
                    "Parity receipt identity mismatch")
            require(receipt["spectral"]["frames"] == row["spectralFrames"] and
                    [case["samples"] for case in receipt["fixtures"]] == row["audioSampleCounts"],
                    "Incomplete streaming reference cases")
            require(receipt["files"], "Empty parity fixture set")
            for filename, record in receipt["files"].items():
                require(re.fullmatch(r"[A-Za-z0-9_.-]+\.f32", filename) is not None and
                        record["file"] == filename, "Invalid parity fixture path")
                expected[f'l{row["layers"]}/{filename}'] = record
        for name, record in catalog["parityMetadataFiles"].items():
            require(name not in expected, "Duplicate parity metadata path")
            expected[name] = record
        require(set(names) == set(expected), "Unexpected or missing parity archive files")
        contents = {}
        for name, record in expected.items():
            data = archive.read(name)
            checked_bytes(data, record, name)
            contents[name] = data
    for name, data in contents.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return len(contents)


def fetch(output, catalog):
    require(catalog["schemaVersion"] == 1 and catalog["releaseTag"] == RELEASE_TAG and
            catalog["supportedLayers"] == list(range(3, 13)) and
            [row["layers"] for row in catalog["models"]] == list(range(3, 13)) and
            catalog["precision"] == "float32" and catalog["license"] == "NSCLv1" and
            catalog["researchOnly"] is True, "Invalid trusted streaming catalog")
    require(not output.exists(), "Use a fresh fixture output directory")
    output.mkdir(parents=True)
    models = output / "models"
    models.mkdir()
    for row in catalog["models"]:
        download(asset_record(row, "model"), models)
        path = download(asset_record(row, "manifest"), models)
        manifest = json.loads(path.read_bytes())
        require(manifest["modelSha256"] == row["modelSha256"] and manifest["layers"] == row["layers"] and
                manifest["checkpointSha256"] == catalog["checkpointSha256"], "Model manifest identity mismatch")
    archive = download(catalog["parityArchive"], output)
    count = extract_parity(archive, output / "parity", catalog)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--github-env", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    count = fetch(output, json.loads(CATALOG.read_text()))
    if args.github_env:
        require("\n" not in str(output) and "\r" not in str(output), "Invalid environment path")
        with args.github_env.open("a") as stream:
            stream.write(f"REUSE_STREAMING_ASSETS={output}\n")
    print(f"Verified ten FP32 streaming models/manifests and {count} synthetic fixture/notice files")


if __name__ == "__main__":
    main()
