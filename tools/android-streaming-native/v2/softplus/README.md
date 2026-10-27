# Stable packet Softplus V2

This separate ARM64 candidate retains the inline-decay V2 frequency/temporal
kernels and replaces the 24 scalar ORT Softplus operations with a bounded
packet implementation from ORT's exact pinned Eigen revision.

ORT's original Eigen Select chooses its scalar branch lazily; the optimization
does **not** assume duplicated branch evaluation. Select lacks packet support
and NEON advertises `HasLog1p=0`, preventing normal expression vectorization.
The new kernel explicitly evaluates `max(x,0) + log1p(exp(-abs(x)))` in four-wide
FP32 packets. It uses Eigen's corrected `generic_log1p`, preserving small
arguments rather than computing naive `log(1+x)`. Any packet containing a
nonfinite input or `abs(x)>80` uses ORT's selected scalar formula, preserving
subnormal and edge behavior. Ordinary FP contraction remains disabled.

All eight layers, original FP32 weights, 31 cache schemas, zero lookahead,
STFT/overlap behavior, and baseline artifact identities remain unchanged. The
V1, bulk V2, and inline V2 variants are independent frozen fallbacks.

Build and validate from the repository root:

```sh
python tools/android-streaming-native/v2/softplus/transform_stable_v2.py
python tools/android-streaming-native/v2/softplus/build_stable_v2.py
python tools/android-streaming-native/v2/softplus/validate_final_exp.py
python tools/android-streaming-native/v2/softplus/validate_softplus.py
python tools/android-streaming-native/v2/softplus/test_stable_v2.py
python tools/android-streaming-native/v2/softplus/validate_stable_v2.py
python tools/android-streaming-native/v2/softplus/build_stable_v2.py --android-abi arm64-v8a
```

Graph: `reuse-streaming-l8-native-stable-v2.onnx`; library:
`libreuse_streaming_stable_v2.so`; custom domain `ai.reuse.streaming`, ABI2.
Only the separate strict stable V2 factory may load this candidate. Retain its
SessionOptions until its session closes. Build receipts verify both C exports,
source hashes, and every vendored Eigen file against the archive provenance.

Numerical proof covers the final library's actual Exp/state kernel, raw
Softplus range/edge/subnormal cases, all carried model states, exact reset,
seven synthetic waveform lengths, and the independent pinned CPU oracle.
Changes stay within measured FP32 rounding differences; bitwise equality with
the original complete graph is not asserted. Host timings cannot establish
physical Android latency or broad listening quality.

Eigen is unmodified under its included upstream notices and MPL-2.0 terms;
MLAS retains its MIT notice in the inline variant. NVIDIA model and derived
inference code retain NSCLv1.
