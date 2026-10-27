#!/usr/bin/env python3
"""Build both native Vulkan targets and record exact source/toolchain identities."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PINNED_NDK = "27.2.12479018"


def digest(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def execute(command: list[str], env: dict[str, str]) -> str:
    process = subprocess.run(command, env=env, text=True, capture_output=True)
    print(process.stdout, end="")
    if process.returncode:
        print(process.stderr, end="")
        raise RuntimeError(f"Build command failed ({process.returncode}): {command[0]}")
    return process.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--host-only", action="store_true")
    targets.add_argument("--android-only", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "generated")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8:
        raise ValueError("Use between one and eight build workers")
    sdk = Path(os.environ.get("ANDROID_HOME", str(Path.home() / "Library/Android/sdk")))
    ndk = sdk / "ndk" / PINNED_NDK
    cmake_candidate = sdk / "cmake/3.22.1/bin/cmake"
    cmake = str(cmake_candidate) if cmake_candidate.is_file() else shutil.which("cmake")
    if not cmake:
        raise FileNotFoundError("Install CMake 3.22 or newer")
    env = os.environ.copy()
    jdk = Path("/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home")
    if jdk.is_dir():
        env["JAVA_HOME"] = str(jdk)
    generated = args.output.resolve()
    generated.mkdir(parents=True, exist_ok=True)
    record = {
        "schemaVersion": 1,
        "createdUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"system": platform.system(), "machine": platform.machine()},
        "compilerFlags": ["-O3", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"],
        "cmakeVersion": execute([cmake, "--version"], env).splitlines()[0],
        "sources": {},
        "targets": {},
    }
    for path in sorted((ROOT / "native").iterdir()):
        if path.is_file() and path.suffix in {".cpp", ".h", ".txt", ".py", ".md"}:
            record["sources"][str(path.relative_to(ROOT))] = digest(path)
    record["sources"]["build_native.py"] = digest(Path(__file__).resolve())
    selected = (["android"] if args.android_only else ["host"] if args.host_only else ["host", "android"])
    for target in selected:
        build = generated / ("build-host" if target == "host" else "build-arm64-v8a")
        configure = [cmake, "-S", str(ROOT / "native"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"]
        target_record = {"commands": [], "artifacts": {}}
        if target == "android":
            source_properties = ndk / "source.properties"
            if not source_properties.is_file() or f"Pkg.Revision = {PINNED_NDK}" not in source_properties.read_text():
                raise FileNotFoundError(f"Required Android NDK {PINNED_NDK} was not found at {ndk}")
            configure += [f"-DCMAKE_TOOLCHAIN_FILE={ndk / 'build/cmake/android.toolchain.cmake'}",
                          "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-26"]
            target_record["ndkVersion"] = PINNED_NDK
            target_record["ndkSourceProperties"] = digest(source_properties)
        target_record["commands"].append(configure)
        execute(configure, env)
        command = [cmake, "--build", str(build), "-j", str(args.jobs)]
        target_record["commands"].append(command)
        execute(command, env)
        library = build / ("libreuse_streaming_vulkan.so" if target == "android" else "libreuse_streaming_vulkan.dylib")
        if not library.is_file():
            raise FileNotFoundError("JNI library was not produced; a host JDK is required")
        target_record["artifacts"]["library"] = digest(library)
        target_record["artifacts"]["cli"] = digest(build / "reuse-vulkan-cli")
        compile_file = build / "CMakeFiles/reuse_vulkan_core.dir/flags.make"
        target_record["actualCompileFlags"] = compile_file.read_text()
        if target == "android":
            readelf = ndk / "toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-readelf"
            dynamic = execute([str(readelf), "--dynamic", str(library)], env)
            needed = [line.strip() for line in dynamic.splitlines() if "NEEDED" in line]
            if any("onnx" in line.lower() for line in needed):
                raise RuntimeError("Unexpected ONNX Runtime dependency in Vulkan engine")
            target_record["sharedLibraryDependencies"] = needed
            symbols = execute([str(readelf), "--dyn-syms", "--wide", str(library)], env)
            target_record["jniExports"] = [line.split()[-1] for line in symbols.splitlines()
                                          if " Java_io_riftco_reusestreaming_NativeVulkanEngine_" in line]
            if len(target_record["jniExports"]) != 11:
                raise RuntimeError("Expected eleven exported Vulkan JNI methods")
        record["targets"][target] = target_record
    receipt = generated / ("native-build-receipt.json" if len(selected) == 2 else f"native-build-{selected[0]}-receipt.json")
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"receipt": str(receipt), "libraries": {
        key: value["artifacts"]["library"] for key, value in record["targets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
