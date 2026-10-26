# Specialized streaming Metal kernels

The package offers an explicit FP32 execution option for eight- and twelve-layer
Real-time RE-USE. It preserves the weights, all selected layers, recurrent states,
16 kHz framing and zero lookahead. The existing standard path remains the default.

```swift
import ReuseStreaming

let enhancer = try StreamingEnhancer(
    modelURL: checkpointURL,
    layers: 12, // 8 is also supported
    optimization: .specializedMetal)
try enhancer.warmUp()
let available = try enhancer.push(mono16kSamples)
let tail = try enhancer.finish()
```

Use one serial background worker. Do not run inference in an audio callback.
The option requires optimized execution, eight or twelve layers and the tested
Apple SIMD32 GPU path. It changes reduction order, so numerical tolerance rather
than bit identity is the appropriate baseline comparison.

## Implementation

- `MambaParallelTemporal` assigns sixteen lanes to the sixteen components of
  each channel's recurrent state and uses a balanced output reduction.
- `MambaPreparedFrequency` prepares delta and the gate once per frequency/channel.
  This path uses separate frequency branches rather than the standard pooled scan.
- `StreamingNormalizationFusion` combines 64-channel LayerNorm with PReLU or a
  residual epilogue, preserving centered variance, learned parameters and epsilon.
- The compiled graph carries all caches as explicit inputs/outputs. Reset and
  replay retain the existing stream contract.

The optional precomputed-decay experiment and command-buffer scheduling changes
were not selected. No quantization, layer pruning, frame skipping or training is
part of this implementation.

## Measurements and scope

The [technical notes](technical-notes.md) describe historical **research iPhone
build** measurements. They are not new phone benchmarks of this extracted package.
Eight-layer matched means decreased from 27.649 to 13.012 ms, and twelve-layer
means from 39.958 to 17.197 ms. Reverse-order tests confirmed the improvement.
Output differences were at FP32 rounding scale on the paired diagnostic.

The twelve-layer repeated minute averaged 16.977 ms with 7/3,001 frames above
20 ms. Full call latency, varied-input quality, battery and every-device behavior
remain unmeasured. Long comparisons have differing thermal histories.

Read the [detailed technical report](REPORT.md) for
the equations, profiling, ablations, latency distributions and validation limits.
[Measurements](measurements.json), [validation](validation.json),
[sustained behavior](sustained.json) and [provenance](provenance.json) contain
sanitized results and source hashes. Private source records are not distributed;
hashes provide identity, not full independent reproducibility.

## Package validation

The extracted package passed **37 macOS Release tests**, including checkpoint-backed
GPU, recurrent-state, reset and chunked-stream checks at both depths. Three optional
timing benchmarks were skipped; no correctness tests failed. The unsigned generic
iOS Release build also passed. See [package validation](package-validation.json)
for commands, source fingerprints, test scope and the distinction from phone timing.

The streaming model and derivative port remain **noncommercial research and
education only** under [NVIDIA NSCLv1](../../apple/Research/LICENSE.txt).
