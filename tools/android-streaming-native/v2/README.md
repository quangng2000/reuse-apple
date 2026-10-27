# Experimental native V2

This separate opt-in eight-layer FP32 graph and ABI2 library preserve the V1
frequency recurrence and fuse eight temporal state updates, reductions, skip
connections, and gates. The graph retains ONNX Runtime's prepared vectorized
Exp and gate operators. All original weights, 31 cache schemas, hop behavior,
and zero lookahead remain unchanged. The V1 artifacts are never overwritten.

Run from the repository root using the same pinned Python dependencies as V1:

```sh
python tools/android-streaming-native/v2/transform_v2.py
python tools/android-streaming-native/v2/build_v2.py
python tools/android-streaming-native/v2/build_v2.py --android-abi arm64-v8a
```

Library: `libreuse_streaming_native_v2.so`; ABI2 entry point and strict ORT1.23.2
registration. Custom domain `ai.reuse.streaming` contains `FrequencyScanV1`
(unchanged frequency operator ABI1) and `TemporalStepV2` (ABI2). Only an explicit
V2 factory may load this graph and its independently hash-verified library.
Keep SessionOptions alive until the corresponding session is closed.

Model/derived inference code retain NVIDIA NSCLv1. ONNX Runtime headers retain
their MIT license from the parent experimental directory. This is a private
experimental optimization, with no claim of physical-device latency or broad
audio quality until validation and measurement complete.
