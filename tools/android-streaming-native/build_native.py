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
    files = ["frequency_scan.cpp", "CMakeLists.txt", "build_native.py", "transform.py", "vendor/onnxruntime-1.23.2/provenance.json"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


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
    output = build / ("libreuse_streaming_scan" + suffix)
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
    receipt = {"kernelAbi": 1, "domain": "ai.reuse.streaming", "op": "FrequencyScanV1", "ortVersion": "1.23.2",
               "abi": abi, "file": output.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
               "sourceSha256": hashlib.sha256((ROOT / "frequency_scan.cpp").read_bytes()).hexdigest(),
               "sources": source_receipt(),
               "verifiedExportedSymbols": exports,
               "compilerFlags": "-O3 -fno-fast-math -ffp-contract=off -fvisibility=hidden", "precision": "float32"}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
