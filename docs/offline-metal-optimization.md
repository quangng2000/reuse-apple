# Offline Metal optimization experiment

The streaming optimization ideas were adapted to the **full 30-layer offline
RE-USE model** on 2026-10-26. They preserved numerical output, but did **not**
produce a reliable speedup on the tested Apple M4 Pro. The original implementation
remains the default. No app deployment, checkpoint, training process, or published
release was changed by this experiment.

## What was implemented

- **SIMD16 scan:** sixteen lanes cooperate on each channel's Mamba state. Delta
  softplus and the SiLU gate are prepared once, then the kernel traverses the
  entire sequence. It preserves every state component and both directions.
- **Normalization fusion:** one kernel computes the 64-channel Mamba LayerNorm,
  its affine transform and the outer residual addition. Encoder/decoder instance
  normalization remains unchanged.
- **Prepared serial scan:** shares the delta/gate preparation but retains the
  original channel-parallel scan and sequential state summation.
- **Scheduling-only variant:** removes the synchronous `eval(y)` between scan
  and output projection. Other evaluation boundaries remain in place.

All candidates use the same pinned checkpoint, FP32, all 30 blocks, original
STFT/inverse STFT, and existing four-second windows with 0.5-second crossfades.
There is no quantization, layer removal, retraining, or reduced context relative
to the existing Apple offline engine. These independent implementations derive
from the offline equations; no NSCL streaming implementation is imported into
the offline product.

## Measurements and decision

Each comparison ran the same Release executable and checkpoint on the same
four-second, 16 kHz mono damaged recording. Each process performed three full
inferences; the first was excluded from the warmed result. Modes ran in forward
then reverse order. Each mean contains four warmed observations. Measurements
include analysis, all model layers, synthesis and GPU completion; checkpoint
loading and file I/O are excluded.

The first comparison measured **2.284 s** for standard, **2.660 s** for SIMD16
scan only, **2.243 s** for normalization only and **2.639 s** for their combination.
The combined candidate took **15.6% longer** than standard. Normalization's
aggregate 1.8% reduction was not consistent across ordering: its reverse-order
mean was 2.251 s versus 2.235 s for standard. This is not a demonstrated gain.

Two separate same-binary comparisons measured **2.654 s standard / 2.673 s
prepared serial**, and **2.313 s standard / 2.351 s scheduling only**. Their
baselines differ; compare within each experiment, not across experiments.
The prepared-serial comparison also showed substantial timing drift.

All recorded thermal states were nominal. This was an active desktop, not an
isolated performance lab; background system services were present. These short
tests neither establish universal performance nor predict iPhone latency.
MLX reported about **806.8 MiB** process peak allocation in the first comparison,
including loading and all three inferences; this is not total application memory.
Prepared step/gate tensors add about **63.2 MiB**
of intermediate storage at the four-second scan shape, even though the reported
process peak was unchanged in that comparison.

**Decision:** retain the standard execution path. Keep the candidates explicitly
experimental for reproducible device measurements, without enabling them in the
offline app or advertising the streaming speedup for the offline model.

## Audio preservation

On this paired diagnostic, input STOI was **0.500000007** and the standard output
STOI was **0.781481946**, with SI-SDR **6.985625 dB**. Across candidates, the largest
STOI difference was **9.58e-9**, and the largest SI-SDR difference was **2.19e-7 dB**.
The largest absolute sample difference among the
first three candidates was **1.49e-7**, and relative RMS error stayed below
**4.93e-7**. Prepared serial and scheduling-only outputs were bit-identical to
their standard references. The new default was also bit-identical to the
pre-change default.

These are numerical-preservation checks on one recording, not a new model-quality
study or listening-test result. Raw aligned samples were compared without gain
matching, normalization, trimming or resampling. The diagnostic originates from
LibriSpeech `1272-128104-0000` (CC BY 4.0), as described in
[the existing fixture attribution](streaming-validation.md).

The final package suite passed **48 tests**, with zero failures and three skipped
optional streaming microbenchmarks. Offline checks include an independent scalar
scan oracle, extreme delta biases, varied sequence lengths, all 30 blocks,
intermediate features, spectrum and waveform agreement, independent repeated
calls, odd sample counts, and a 64,001-sample crossfade boundary. The unsigned
iOS library build also passed; this does not establish physical-iPhone latency.

## Why the streaming result did not transfer

The offline baseline already stores all sixteen recurrent state values locally
within each thread and processes a complete sequence. It has many independent
batch/channel threads. Moving state to sixteen cooperating lanes introduces
per-position shuffle reductions and duplicated loads; preparation adds memory
traffic and another dispatch. These costs are a plausible explanation for the
observed regression, not a profiler-proven attribution. The offline workload
requires its own profiling rather than extrapolation from one-frame streaming.

## Try the experiment

The experimental API is on this development branch, not the existing `v0.2.0` tag:

```swift
// Benchmark only; standard remains recommended for normal use.
let engine = try ReuseEngine(modelURL: checkpointURL, optimization: .specializedMetal)
```

The CLI additionally exposes `scan-only`, `normalization-only`, `prepared-serial`
and `scheduling-only` ablations. Build `ReuseProbe` using the
[Xcode instructions](development.md), then run:

```sh
python3 apple/scripts/benchmark_offline_metal.py \
  --binary BuildPackage/Build/Products/Release/ReuseProbe \
  --model models/model.safetensors --input diagnostic.wav \
  --output-dir Validation/offline-ablation
```

Python 3.11+ is required. Use an empty output directory. To reproduce the two
additional comparisons, use a new directory and append respectively
`--modes standard prepared-serial` or `--modes standard scheduling-only`.
The script records timings, thermal states, allocation, sample counts, and
SHA-256 hashes without downloading weights or changing remote resources.

For aligned paired audio, install `numpy`, `soundfile`, and `pystoi` in your test
environment and run `apple/scripts/compare_offline_audio.py --help`. Pass the clean,
damaged and standard-output WAVs plus the ablation output directory. It saves
STOI, SI-SDR, sample-error and hash evidence without altering the audio.

See [sanitized measurement evidence](benchmarks/offline-metal-20261026.json).
Model weights, recordings, machine paths and build products are excluded from Git.
