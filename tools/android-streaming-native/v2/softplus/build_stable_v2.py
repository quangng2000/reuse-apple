#!/usr/bin/env python3
"""Compile only the small custom-op library, never the ONNX Runtime itself."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent

def android_sdk():
    configured = os.environ.get("ANDROID_SDK_ROOT") or os.environ.get("ANDROID_HOME")
    return Path(configured) if configured else Path.home() / "Library/Android/sdk"

def source_receipt():
    files = ["native_stable_v2.cpp", "CMakeLists.txt", "build_stable_v2.py", "transform_stable_v2.py", "stable_softplus.cpp", "vendor/eigen/provenance.json"]
    eigen = ROOT / "vendor/eigen"
    for name, expected in json.loads((eigen / "provenance.json").read_text())["files"].items():
        actual = (eigen / name).read_bytes()
        assert len(actual) == expected["bytes"] and hashlib.sha256(actual).hexdigest() == expected["sha256"], name
    result = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    result.update(inlineMlasHeader=hashlib.sha256((ROOT.parent / "inline/mlas_exp_neon.h").read_bytes()).hexdigest(),
                  inlineMlasProvenance=hashlib.sha256((ROOT.parent / "inline/mlas-provenance.json").read_bytes()).hexdigest(),
                  v1FrequencySource=hashlib.sha256((ROOT.parent.parent / "frequency_scan.cpp").read_bytes()).hexdigest(),
                  v1TransformSource=hashlib.sha256((ROOT.parent.parent / "transform.py").read_bytes()).hexdigest(),
                  ortHeaderProvenance=hashlib.sha256((ROOT.parent.parent / "vendor/onnxruntime-1.23.2/provenance.json").read_bytes()).hexdigest())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--android-abi", choices=["arm64-v8a", "armeabi-v7a", "x86", "x86_64"])
    parser.add_argument("--ndk", type=Path, default=android_sdk() / "ndk/27.2.12479018")
    parser.add_argument("--cmake", default=shutil.which("cmake") or str(android_sdk() / "cmake/3.22.1/bin/cmake"))
    args = parser.parse_args()
    abi = args.android_abi or "host"
    build = ROOT / "generated" / ("build-" + abi)
    command = [args.cmake, "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
    if args.android_abi:
        command += ["-DCMAKE_TOOLCHAIN_FILE=" + str(args.ndk / "build/cmake/android.toolchain.cmake"),
                    "-DANDROID_ABI=" + abi, "-DANDROID_PLATFORM=android-26", "-DANDROID_STL=c++_static"]
    subprocess.run(command, check=True)
    subprocess.run([args.cmake, "--build", str(build), "--parallel", "2"], check=True)
    suffix = ".so" if args.android_abi or platform.system() != "Darwin" else ".dylib"
    output = build / ("libreuse_streaming_stable_v2" + suffix)
    assert output.is_file()
    if args.android_abi:
        llvm_bins = list((args.ndk / "toolchains/llvm/prebuilt").glob("*/bin/llvm-nm"))
        assert len(llvm_bins) == 1, "Pass an NDK with exactly one host llvm-nm"
        symbol_command = [str(llvm_bins[0]), "-D", "--defined-only", str(output)]
    elif platform.system() == "Darwin":
        symbol_command = ["/usr/bin/nm", "-gU", str(output)]
    else:
        symbol_command = ["nm", "-D", "--defined-only", str(output)]
    symbols = {line.split()[-1].lstrip("_") for line in subprocess.check_output(symbol_command, text=True).splitlines() if line.split()}
    exports = ["RegisterCustomOps", "ReuseStreamingKernelAbiVersion"]
    assert all(symbol in symbols for symbol in exports), "Custom-op C entry points must be dynamically exported"
    data = output.read_bytes()
    receipt = {"kernelAbi": 2, "domain": "ai.reuse.streaming", "op": "FrequencyScanV2", "temporalOp": "TemporalStepV2", "softplusOp": "StableSoftplusV2", "ortVersion": "1.23.2",
               "abi": abi, "file": output.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
               "sourceSha256": hashlib.sha256((ROOT / "native_stable_v2.cpp").read_bytes()).hexdigest(),
               "sources": source_receipt(),
               "verifiedExportedSymbols": exports,
               "compilerFlags": "-O3 -fno-fast-math -ffp-contract=off -fvisibility=hidden", "precision": "float32"}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
