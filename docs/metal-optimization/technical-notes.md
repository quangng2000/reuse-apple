# Specialized FP32 Metal kernels for streaming RE-USE

These measurements were collected on 2026-10-26 from the historical research
application on an iPhone 13 Pro Max / A15 running iOS 26.2.1. They are **not phone
benchmarks of the newly extracted public package**. Package build and parity
checks must be reported separately; these results do not establish its speed.

## What changed

Three execution optimizations were enabled together: prepared frequency delta
and gate, fused normalization, and parallel temporal recurrence. Both sides
used the same checkpoint, selected eight- or twelve-layer exit, FP32 arithmetic,
16 kHz mono input, zero lookahead, 640-sample window and 320-sample hop. Existing
filter/projection fusion and whole-frame compilation remained enabled on both
sides. There was no training, quantization, frame skipping or depth reduction.

For channel `c` and state component `n`, the Mamba step retains these equations:

```text
delta[c] = softplus(sum(r=0..3, projected[r] * W[c,r]) + bias[c])
state'[c,n] = exp(delta[c] * A[c,n]) * state[c,n]
              + delta[c] * B[n] * u[c]
y[c] = (sum(n=0..15, C[n] * state'[c,n]) + D[c] * u[c]) * SiLU(z[c])
```

The temporal kernel assigns sixteen adjacent lanes to the sixteen state
components of one channel. A leader computes delta once and broadcasts it;
each component writes its next state exactly once. A four-stage reduction forms
the output. Two independent channels share the tested SIMD32 group. Persistent
temporal state is carried across frames; the output sum changes association.

The frequency path prepares delta and gate once per frequency/channel, avoiding
their repetition in sixteen component lanes. Its recurrence remains sequential
across frequency, with zero frequency state at each frame. The selected path
does not use an associative prefix replacement or precompute all decay terms.

Normalization fuses fixed 64-channel LayerNorm with its PReLU or residual
epilogue. It retains centered variance, epsilon `1e-5`, learned scale and bias.
SIMD reductions change floating-point summation order. These specialized kernels
assume the tested 256-channel, sixteen-component and Apple SIMD32 contracts.

## Controlled short comparisons

Each pair used the same Release binary, one four-second diagnostic, 201 frames
including startup/EOF, nominal thermal snapshots and Low Power Mode off.
The operator recorded sequential runs without concurrent inference or profiling.
Loading and three warmup frames were excluded. Order was reversed for pair B.

- Eight layers, pair A: **27.649 → 13.012 ms mean**, **28.868 → 14.253 ms p95**.
  Pair B: **28.062 → 13.094 ms mean**, **30.065 → 14.914 ms p95**.
  Mean reductions were **52.94% / 53.34%**; optimized misses were **0/201 / 1/201**.
- Twelve layers, pair A: **39.958 → 17.197 ms mean**, **41.898 → 18.923 ms p95**.
  Pair B: **38.920 → 17.194 ms mean**, **41.474 → 19.150 ms p95**.
  Mean reductions were **56.96% / 55.82%**; optimized misses were **4/201 / 6/201**.

These are processing times for analysis, network, synthesis and GPU completion,
not complete call latency or a guarantee that every frame meets the 20 ms hop.

## Sustained replay and numerical checks

The long tests repeat the same clip fifteen times with continuous state:
960,000 samples and 3,001 frames. Repetition extends duration, not diversity.
Eight-layer optimized processing averaged **14.036 ms**, p95 **14.772 ms**, maximum
**18.761 ms**, with **0/3,001** misses; 60 seconds processed in **42.353 seconds**.
It started/ended serious, after a baseline run changed nominal → serious.
Twelve-layer optimized processing averaged **16.977 ms**, p95 **17.270 ms**, maximum
**26.570 ms**, with **7/3,001** misses; 60 seconds processed in **51.135 seconds**.
It ran first, nominal → fair; the baseline then changed fair → serious.
Both long pairs include six allocation snapshots and are **thermally confounded**;
their speed ratios are not controlled comparisons. Sampled MLX allocations showed
no monotonic growth; this is neither process RSS measurement nor a leak test.

All phone outputs were finite. Raw waveform comparisons used no gain adjustment
or alignment. Short relative RMS errors were **−123.08 dB** at eight layers and
**−125.05 dB** at twelve. Paired short STOI was **0.7182996299 → 0.7182996319** and
**0.7188348065 → 0.7188348030**, respectively. Long STOI was
**0.7242358753 → 0.7242358903** and **0.7222435139 → 0.7222435174**.
All specified waveform, STOI and SI-SDR gates passed on this diagnostic.

Separate four-second Mac probes checked every state tensor at frames
0, 1, 50, 100, 150 and 200: **31 tensors at eight layers, 39 at twelve**.
FP32 shape/finite checks and per-tensor error gates passed. Each implementation
replayed its own waveform and sampled state hashes exactly after reset.
Phone intermediate states and all sixty-second states were not directly checked.
These results support numerical preservation on this input, not general speech
quality, other GPUs, thirty-layer streaming, or microphone-to-speaker latency.

## Evidence, provenance and license

[Measurements](measurements.json), [state validation](validation.json),
[sustained windows and allocations](sustained.json), and [provenance](provenance.json)
retain exact values and source SHA-256 identities. Original reports remain private;
hashes alone do not make the experiment fully reproducible. Audio, weights, traces,
device identifiers and absolute local paths are excluded. See the existing
[diagnostic attribution](../streaming-validation.md#evidence-and-attribution).
The streaming model and derivative port retain the **noncommercial research and
education** terms of [NVIDIA NSCLv1](../../Research/LICENSE.txt); the offline
component's root license does not replace them. See [NOTICE](../../NOTICE).
