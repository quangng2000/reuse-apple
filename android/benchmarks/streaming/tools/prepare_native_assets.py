#!/usr/bin/env python3
"""Stage only the validated, explicit L8 native-frequency experiment."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[4]
APP = Path(__file__).resolve().parents[1] / "app"
GENERATED = ROOT / "tools/android-streaming-native/generated"
MODEL_SHA = "6b6cbd4117146f3137c1a0db71ecf04c23b8ba97bab7d973ece71c8739aa7756"
BASELINE_SHA = "18db8d32b638702909de484d2a9cffc97ed09e6454c4caed3f0008fb0aab53e8"


def main():
    validation = json.loads((GENERATED / "validation.json").read_text())
    manifest_path = GENERATED / "reuse-streaming-l8-native-frequency-v1.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    receipt_path = GENERATED / "build-arm64-v8a/libreuse_streaming_scan.so.json"
    receipt = json.loads(receipt_path.read_text())
    assert validation["passed"] is True and validation["nativeModelSha256"] == MODEL_SHA
    assert validation["allOriginalInitializerBytesUnchanged"] is True
    assert validation["allExternalInputOutputSchemasUnchanged"] is True
    assert manifest["modelSha256"] == MODEL_SHA and manifest["baselineModelSha256"] == BASELINE_SHA
    assert manifest["executionVariant"] == "native-frequency-scan-v1" and manifest["layers"] == 8
    assert manifest["nativeSources"] == receipt["sources"] and receipt["abi"] == "arm64-v8a"
    assert receipt["ortVersion"] == "1.23.2" and receipt["precision"] == "float32"
    model = GENERATED / manifest["modelFile"]
    library = receipt_path.parent / receipt["file"]
    assert model.stat().st_size == manifest["modelBytes"] and hashlib.sha256(model.read_bytes()).hexdigest() == MODEL_SHA
    assert library.stat().st_size == receipt["bytes"] and hashlib.sha256(library.read_bytes()).hexdigest() == receipt["sha256"]
    assets = APP / "build/benchmark-native-assets/native"
    libs = APP / "build/benchmark-native-libs/arm64-v8a"
    assets.mkdir(parents=True, exist_ok=True)
    libs.mkdir(parents=True, exist_ok=True)
    for source in (model, manifest_path, receipt_path):
        shutil.copyfile(source, assets / source.name)
    shutil.copyfile(library, libs / library.name)
    print("Staged validated L8 FP32 native graph and exact arm64 kernel:", receipt["sha256"])


if __name__ == "__main__":
    main()
