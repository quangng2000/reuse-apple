# Streaming validation

The following measurements were collected from the optimized research prototype
before extraction into the standalone package. They describe the same pinned
four-layer model and kernel implementation; package build/parity checks are
recorded separately in [development](development.md).

## iPhone processing measurements

Device: **iPhone 13 Pro Max / A15**, iOS 26.2.1, Release build, MLX Swift 0.29.1,
FP32, four layers and zero lookahead. The unchanged four-second 16 kHz crackling
diagnostic was concatenated 15 times into one 60-second stream. State persisted
across all repetitions. Repetitions extend duration, not sample diversity.

- Baseline four-layer implementation: **44.32 ms/frame mean**, **53.07 ms p95**
  on the original four-second test.
- Optimized 60-second replay: **18.28 ms mean**, **19.03 ms p95**, **20.01 ms p99**.
- Maximum: **23.01 ms**; **31/3,001 frames (1.03%)** exceeded 20 ms.
- Processing wall time: **54.97 seconds**; real-time factor **0.916**.
- Six 10-second window means remained between **18.16 and 18.50 ms**.
- Thermal state was nominal and Low Power Mode off at the start/end snapshots.

Timing includes incremental analysis, network, synthesis and GPU completion.
Model loading and three warmup frames are excluded. This was file replay as fast
as possible, not a paced call. No microphone callback, acoustic echo cancellation,
codec, network, jitter-buffer or speaker delay was measured.

A separate short run of the final prototype had a **1.461-second stall at frame
78**, increasing its mean to 25.80 ms. The cause was not established. It is
retained in the evidence and did not recur in the longer test. Results are not
a guarantee that every frame, device or full call will meet a 20 ms deadline.

## Numerical preservation

The speedup came from parallel Mamba state calculations, fused filtering and
projection kernels, and one evaluation after the network and its states. The
checkpoint, layer count, FP32 precision and audio transforms were unchanged.

- Short iPhone output versus original implementation: waveform RMSE **2.01e-8**,
  maximum absolute sample difference **2.09e-7**.
- Paired diagnostic STOI: **0.669841891 → 0.669841881**.
- Official sequential spectrum checks covered exits 4/12 and lookahead 0/2;
  maximum magnitude RMSE was no larger than **1.9e-8**.
- A 60-second Mac waveform comparison had RMSE **2.34e-8** and maximum difference
  **3.20e-7**, with no material error growth over six 10-second windows.

These checks establish numerical preservation on the tested inputs, not general
speech quality. Small floating-point reorderings remain. A short microphone
comparison also matched to rounding-level differences, but it had no clean
reference and its personal recording is not distributed.

## Evidence and attribution

[Sanitized benchmark summary](benchmarks/iphone13-streaming.json) includes the
model/input hashes, timing distribution, finite-output checks and the earlier
outlier. Device identifiers, absolute local paths and personal recordings are
excluded. The diagnostic audio is not bundled with the package.

The diagnostic source was the first four seconds of LibriSpeech utterance
`1272-128104-0000`, with synthetic crackling calibrated to input STOI 0.5000000067.
LibriSpeech is by Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur,
distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) via
[OpenSLR SLR12](https://www.openslr.org/12). Its modified WAV SHA-256 is recorded
in the summary. Use your own audio with the public CLI to benchmark your device.
