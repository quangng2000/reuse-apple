# Experimental Android streaming frequency kernel

This is an explicit experimental eight-layer CPU backend for the released
`reuse-streaming-l8-fp32.onnx` graph. It does not replace the published model,
manifest, exporter, or SDK default. It requires ONNX Runtime **1.23.2** and
custom-op ABI **1**. NVIDIA model and derived inference code retain NSCLv1;
the vendored ONNX Runtime headers retain their MIT license.

The baseline graph must have SHA-256
`18db8d32b638702909de484d2a9cffc97ed09e6454c4caed3f0008fb0aab53e8`.
The transform refuses other sources and asserts that every original initializer
and external input/output schema remains unchanged. All eight layers, FP32
weights, temporal caches, zero lookahead, FFT, and streaming overlap behavior
remain present.

## What is fused

The source graph already computes delta projection and Softplus outside each
frequency Scan. This variant also prepares gate and vectorized decay outside
the loop, then replaces each of the **16 frequency Scans** with one bounded
`ai.reuse.streaming:FrequencyScanV1` CPU kernel. The kernel updates each of the
16 state components in FP32 and preserves serial recurrence over 161 frequency
bins. Independent channels may use ONNX Runtime's existing intra-op pool.

This applies Apple's prepared-frequency lesson to Android's actual bottleneck:
eliminate per-bin operator dispatch and repeated gate work, keep streaming
state bounded, and parallelize only independent channels. It deliberately uses
ORT's bulk Exp instead of millions of scalar `expf` calls. Compilation disables
fast math and multiply-add contraction. Reduction order differs from ORT's
vectorized ReduceSum, so bitwise equality is not assumed; full state and
unaligned waveform parity must pass before benchmarking.

## Build and validate

Use Python with ONNX 1.19.1, ONNX Runtime 1.23.2, NumPy, and PyTorch. Run from
the repository root:

```sh
python tools/android-streaming-native/transform.py
python tools/android-streaming-native/build_native.py
python tools/android-streaming-native/validate.py
python tools/android-streaming-native/build_native.py --android-abi arm64-v8a
```

`ANDROID_SDK_ROOT`/`ANDROID_HOME`, or `--cmake` and `--ndk`, select Android
build tools. Generated graphs, native binaries, receipts, and validation output
are ignored. Header provenance and SHA-256 hashes are recorded under
`vendor/onnxruntime-1.23.2/`. Native receipts bind the binary to kernel source,
build script, transform, CMake flags, header provenance, runtime, and ABI.

Register the native library with `OrtSession.SessionOptions.registerCustomOpLibrary`
using an absolute `.so` path **before** constructing a session with the
experimental graph. Retain those options until dependent sessions close.
The library uses the ORT API table supplied by the runtime rather than linking
to private Android ORT symbols. A standard ORT session without registration
must reject the custom graph.

An Android benchmark must explicitly opt into both the native library and its
separate graph. Verify all hashes and provenance fields before loading it.
Do not weaken the standard SDK's source or artifact checks to accept this
variant. Host latency is a diagnostic only; an emulator benchmark is also not
a physical Android device measurement.
