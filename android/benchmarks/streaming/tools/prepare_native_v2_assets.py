#!/usr/bin/env python3
"""Stage only the validated, hash-pinned FP32 temporal-fusion experiments."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[4]
APP = Path(__file__).resolve().parents[1] / "app"
BASELINE = "18db8d32b638702909de484d2a9cffc97ed09e6454c4caed3f0008fb0aab53e8"
VARIANTS = [
    ("v2", "native-v2-bulk", "reuse-streaming-l8-native-frequency-temporal-v2",
     "1fbd91be9c90568c58ab40e9ce70e07c54dafc599fffb694a84c1f4d3905b11d",
     "libreuse_streaming_native_v2.so", "3013f39b7e97464b415a6b8b3500be0ac0a26b678ea8d7e71f248178a29b8868"),
    ("v2/inline", "native-v2-inline", "reuse-streaming-l8-native-inline-v2",
     "55d910c30cd7144187e5f358dca6700ae6e7e1051eb8f7758d454f68e9144c11",
     "libreuse_streaming_inline_v2.so", "7a6d98e2f2d13991de0a5aa21aebc8879a262ffdcf2fe85d10a0e027aea3be01"),
    ("v2/softplus", "native-v2-stable", "reuse-streaming-l8-native-stable-v2",
     "db29c725c70608920e600787fcda625e98429cfbae2c2d09f1dc73726825319a",
     "libreuse_streaming_stable_v2.so", "0ef1e0447d0dc64bff5ae588e64cb2c4f3072bd0a6f54cd5c05cb089cb540c0d"),
]


def main():
    for relative, destination, stem, model_sha, library_name, library_sha in VARIANTS:
        generated = ROOT / "tools/android-streaming-native" / relative / "generated"
        manifest_file = generated / f"{stem}.manifest.json"
        library_file = generated / "build-arm64-v8a" / library_name
        receipt_file = library_file.with_suffix(".so.json")
        manifest = json.loads(manifest_file.read_text())
        receipt = json.loads(receipt_file.read_text())
        proof = json.loads((generated / "validation.json").read_text())
        assert proof["passed"] and proof["allOriginalInitializerBytesUnchanged"] and proof["allExternalInputOutputSchemasUnchanged"]
        assert proof["nativeModelSha256"] == model_sha and proof["baselineSha256"] == BASELINE
        assert manifest["modelSha256"] == model_sha and manifest["baselineModelSha256"] == BASELINE
        assert manifest["layers"] == 8 and manifest["precision"] == "float32" and manifest["lookahead"] == 0
        assert manifest["nativeSources"] == receipt["sources"]
        assert receipt["kernelAbi"] == 2 and receipt["abi"] == "arm64-v8a" and receipt["ortVersion"] == "1.23.2"
        assert receipt["precision"] == "float32" and receipt["sha256"] == library_sha
        assert "RegisterCustomOps" in receipt["verifiedExportedSymbols"]
        model_file = generated / f"{stem}.onnx"
        assert model_file.stat().st_size == manifest["modelBytes"] and hashlib.sha256(model_file.read_bytes()).hexdigest() == model_sha
        assert library_file.stat().st_size == receipt["bytes"] and hashlib.sha256(library_file.read_bytes()).hexdigest() == library_sha
        if relative.endswith("inline") or relative.endswith("softplus"):
            exp = json.loads((generated / "raw-exp-validation.json").read_text())
            assert exp["count"] >= 2_000_000 and exp["finiteBitwiseEqual"] and exp["maxFiniteUlp"] == 0
            assert exp["infClassificationEqual"] and exp["nanClassificationEqual"]
        if relative.endswith("softplus"):
            activation = json.loads((generated / "raw-softplus-validation.json").read_text())
            assert activation["count"] >= 2_000_000 and activation["maxFiniteUlp"] <= 4
            assert activation["maxFiniteAbs"] <= 5e-7 and activation["maxPositiveRelative"] <= 4e-7
            assert activation["nanClassificationEqual"] and activation["infClassificationEqual"]
            assert activation["positiveClassificationEqual"] and activation["subnormalBitwiseEqual"]
            assert activation["fallbackExtremeBitwiseEqual"]
        assets = APP / "build/benchmark-native-assets" / destination
        libs = APP / "build/benchmark-native-libs/arm64-v8a"
        assets.mkdir(parents=True, exist_ok=True)
        libs.mkdir(parents=True, exist_ok=True)
        for file in (model_file, manifest_file, receipt_file):
            shutil.copyfile(file, assets / file.name)
        shutil.copyfile(library_file, libs / library_name)
        print("Staged verified FP32 candidate", destination, model_sha, library_sha)
    notices = APP / "build/benchmark-native-assets/native-v2/licenses"
    notices.mkdir(parents=True, exist_ok=True)
    native = ROOT / "tools/android-streaming-native/v2"
    eigen = native / "softplus/vendor/eigen"
    shutil.copyfile(native / "inline/MLAS-LICENSE", notices / "MLAS-MIT.txt")
    shutil.copyfile(eigen / "COPYING.MPL2", notices / "Eigen-MPL-2.0.txt")
    (notices / "NOTICE.txt").write_text(
        "The experimental native streaming library contains an ARM64 FP32 exponential "
        "adapted from ONNX Runtime 1.23.2 MLAS, Copyright (c) Microsoft Corporation, "
        "under MIT; see MLAS-MIT.txt. Source: "
        "https://github.com/microsoft/onnxruntime/tree/v1.23.2/onnxruntime/core/mlas/lib\n\n"
        "It also compiles unmodified Eigen Core headers at commit "
        "1d8b82b0740839c0de7f1242a3585e3390ff5f33, governed by MPL 2.0 and their "
        "retained per-file notices; see Eigen-MPL-2.0.txt and the included COPYING files. "
        "Complete corresponding Eigen headers, notices and provenance are included "
        "in Eigen-Source.zip in this directory. Obtain them at no charge by opening "
        "this APK with a ZIP tool and extracting assets/native-v2/licenses/Eigen-Source.zip. "
        "Upstream: https://github.com/eigen-mirror/eigen/tree/1d8b82b0740839c0de7f1242a3585e3390ff5f33\n",
        encoding="utf-8")
    with zipfile.ZipFile(notices / "Eigen-Source.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file in sorted(eigen.rglob("*")):
            if file.is_file():
                assert eigen.resolve() in file.resolve().parents
                item = zipfile.ZipInfo("eigen/" + file.relative_to(eigen).as_posix(), (1980, 1, 1, 0, 0, 0))
                item.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(item, file.read_bytes())
    print("Staged MLAS/Eigen notices and complete corresponding vendored Eigen headers.")


if __name__ == "__main__":
    main()
