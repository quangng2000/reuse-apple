#!/usr/bin/env python3
"""Compile, validate and audit the fixed FP32-only Vulkan compute shaders.

No performance/relaxed-math flags are enabled. The audit rejects 16-bit floats,
RelaxedPrecision and fast-math flags, and records NoContraction decorations
before any package is admitted to a phone test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
KERNELS = ["unary", "binary", "where", "gather", "pad", "scatter", "matmul",
           "conv", "layernorm", "frequency", "temporal", "finite_check",
           "conv_tiled", "matmul_tiled", "concat_cache"]

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def find_tool(name: str, explicit: str | None) -> Path:
    if explicit:
        result = Path(explicit)
    elif shutil.which(name):
        result = Path(shutil.which(name))
    else:
        sdk = Path(os.environ.get("ANDROID_HOME", str(Path.home()/"Library/Android/sdk")))
        candidates = sorted(sdk.glob(f"ndk/*/shader-tools/*/{name}"))
        if not candidates:
            raise SystemExit(f"Cannot find {name}; install Android NDK shader tools or supply --{name}")
        result = candidates[-1]
    if not result.is_file():
        raise SystemExit(f"Missing compiler tool: {result}")
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"generated/shaders")
    parser.add_argument("--math-variant", choices=["strict", "fma"], default="strict")
    parser.add_argument("--gpu-variant", choices=["baseline", "adreno64", "adreno64-fastindex", "adreno64-conv36", "adreno64-conv72", "adreno64-conv72-separated"], default="baseline")
    for name in ["glslc", "spirv-val", "spirv-dis"]:
        parser.add_argument(f"--{name}")
    args = parser.parse_args()
    if args.gpu_variant in ("adreno64-conv36", "adreno64-conv72", "adreno64-conv72-separated") and args.math_variant != "fma":
        parser.error("Dense convolution variants require --math-variant fma.")
    compiler = find_tool("glslc", args.glslc)
    validator = find_tool("spirv-val", args.spirv_val)
    disassembler = find_tool("spirv-dis", args.spirv_dis)
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schemaVersion": 1, "targetEnvironment": "vulkan1.1", "precision": "FP32", "mathVariant": args.math_variant,
        "gpuVariant": args.gpu_variant,
        "relaxedPrecision": False, "float16": False, "compiler": str(compiler),
        "compilerVersion": subprocess.check_output([str(compiler), "--version"], text=True).strip(),
        "commonSourceSha256": digest(ROOT/"shaders/common.glsl"), "kernels": {},
        "includeSources": {str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/"shaders").rglob("*.glsl"))},
    }
    kernels=KERNELS + (["conv_3x3_tiled"] if args.gpu_variant == "adreno64-conv72-separated" else [])
    for kernel in kernels:
        source = ROOT/"shaders"/(kernel+".comp")
        variant_source = ROOT/"shaders"/args.math_variant/(kernel+".comp")
        if variant_source.is_file():
            source = variant_source
        if args.gpu_variant.startswith("adreno64"):
            group_source = ROOT/"shaders/adreno64"/(kernel+".comp")
            if group_source.is_file():
                source = group_source
        gpu_source = ROOT/"shaders"/args.gpu_variant/(kernel+".comp")
        if gpu_source.is_file():
            source = gpu_source
        output = args.output/(kernel+".spv")
        command = [str(compiler), "--target-env=vulkan1.1", "-fshader-stage=compute",
                   "-I", str(ROOT/"shaders"), str(source), "-o", str(output)]
        subprocess.run(command, check=True)
        subprocess.run([str(validator), "--target-env", "vulkan1.1", str(output)], check=True)
        assembly = subprocess.check_output([str(disassembler), str(output)], text=True)
        if "RelaxedPrecision" in assembly or re.search(r"OpTypeFloat\s+16\b", assembly):
            raise SystemExit(f"Rejected reduced precision shader: {kernel}")
        types = set(int(x) for x in re.findall(r"OpTypeFloat\s+(\d+)", assembly))
        if types != {32}:
            raise SystemExit(f"Unexpected float type in {kernel}: {types}")
        forbidden = ("OpCapability Float16", "OpCapability Float64", "FPFastMathMode")
        if any(x in assembly for x in forbidden):
            raise SystemExit(f"Rejected non-FP32 / fast-math capability in {kernel}")
        receipt["kernels"][kernel] = {
            "source": str(source.relative_to(ROOT)),
            "sourceSha256": digest(source), "spirvSha256": digest(output),
            "bytes": output.stat().st_size, "floatTypes": sorted(types),
            "noContractionDecorations": assembly.count("NoContraction"),
            "localSize": [int(x) for x in re.search(r"OpExecutionMode\s+%\w+\s+LocalSize\s+(\d+)\s+(\d+)\s+(\d+)", assembly).groups()],
        }
    destination = args.output/"shader-receipt.json"
    destination.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"validatedKernels": len(kernels), "receipt": str(destination)}, indent=2))

if __name__ == "__main__":
    main()
