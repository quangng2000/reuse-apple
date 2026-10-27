# ARM64 inline-decay V2 experiment

This independent ABI2 graph/library retains all eight layers and all original
FP32 weights. It fuses the frequency and temporal state arithmetic and computes
decay inside each kernel, avoiding full intermediate `delta * A` and `Exp`
tensors. The V1 and bulk V2 graphs, libraries, factories, and receipts remain
separate and unchanged.

The exponential is the pinned ONNX Runtime 1.23.2 MLAS NEON algorithm with its
original constants, range reconstruction, and fused polynomial arithmetic.
Explicit NEON `vfmaq` preserves the shipped ARM64 ORT arithmetic while implicit
FP contraction stays disabled for the original state Mul/Mul/Add boundaries.
No FP16, quantization, fast-math flag, pruning, lookahead, weight change, or
state-layout change is introduced. This experiment intentionally supports
ARM64 NEON only. MLAS source attribution, upstream file hashes, and MIT license
are included; NVIDIA model/derived state code retain NSCLv1.

From the repository root with the pinned ONNX/ORT/PyTorch environment:

```sh
python tools/android-streaming-native/v2/inline/transform_inline_v2.py
python tools/android-streaming-native/v2/inline/build_inline_v2.py
python tools/android-streaming-native/v2/inline/validate_exp.py
python tools/android-streaming-native/v2/inline/test_inline_v2.py
python tools/android-streaming-native/v2/inline/validate_inline_v2.py
python tools/android-streaming-native/v2/inline/build_inline_v2.py --android-abi arm64-v8a
```

Only `libreuse_streaming_inline_v2.so` is an SDK candidate. The separate
`libreuse_streaming_inline_exp_validation` target exposes raw Exp solely for
validation and must not be packaged in the application. Register the candidate
via an explicit, strictly hash-verified inline V2 factory and keep its ORT
SessionOptions alive until the session closes.

The raw Exp proof covers 2,000,008 FP32 range/random/edge inputs and requires
bitwise equality with ORT, including infinity/NaN classification. Whole-model
validation covers all 31 carried caches, exact reset, seven unaligned waveform
lengths, and the independent pinned CPU oracle. Host timings are diagnostics;
they cannot establish physical Android latency or broad listening quality.
