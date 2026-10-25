# Eight-layer FP32 device comparison

## iPhone 13 Pro Max result, October 25, 2026

On a physical iPhone 13 Pro Max (A15, iOS 26.2.1), the two changes reduced
eight-layer mean frame processing time from **28.962 ms to 28.388 ms (1.99%)**
in the continuous phase. P95 fell from 29.687 ms to 29.026 ms; P99 fell from
30.422 ms to 29.380 ms. These are pooled timings from two 32-second trials per
variant in baseline/candidate/candidate/baseline order: 3,202 completed model
frames per variant, with no excluded frames or outliers. Every trial remained
in the nominal thermal state and low power mode was off.

The candidate processed 32 seconds of audio in about 45.50 seconds (RTF 1.422),
versus 46.42 seconds (RTF 1.451) for the baseline. Every frame exceeded the
20 ms input hop in this unpaced replay, so eight layers still do not sustain
real-time processing on this phone. This is a modest improvement in this measured
session, with two trials per variant; it is not an end-to-end call benchmark.

All eight short and continuous trials had **bit-for-bit identical output** to
their phase's baseline. Output lengths, finite samples and the SHA256 of each
copied output file were verified on the host. The short phase's pooled mean was
29.493 ms baseline versus 29.080 ms candidate (1.40% reduction). Its first
candidate trial included an 88.645 ms frame; this outlier remains in the data.
The continuous phase repeats the same four-second diagnostic input eight times
without resetting recurrent state between repetitions.

The baseline is frozen at `bdf56d19f4dbc16fb9ee949d42b70b4b6b5fe43e`.
The tested candidate engine source fingerprint is
`9b823aaf694fd883394b0f6c476dc0219ce2927f02816e9216c95fa946946cb8`;
per-file hashes are in `BenchAssets/candidate-source.json`.
The two changes share identical delta projection/softplus work across each
frequency scan's 16 state lanes and complete GPU evaluation once after synthesis
while materializing every recurrent and convolution state. FP32, layers,
checkpoint, transforms, state updates and reduction order are preserved.
All 17 macOS package tests passed, including exact frozen eight-layer reference
output, eager/deferred state equality and focused GPU kernel edge cases.

Raw timings, output hashes and trial metadata:
[device results](Reports/2026-10-25-iphone13promax-results.json).
Pooled metrics, all trial summaries and limitations:
[validated summary](Reports/2026-10-25-iphone13promax-summary.json).

## Reproduce the comparison

This diagnostic app compares the frozen current optimized streaming implementation
with the candidate implementation. **Both variants use eight layers, FP32,
zero lookahead, the same pinned checkpoint, and `optimized: true`.** The reference
is not the older `optimized: false` implementation.

The parent workspace supplies the immutable reference source under
`../Reference/Sources/ReuseStreaming` and verifies it against the recorded baseline
revision. The reference is a private static Xcode framework target, not a new
public library product. Its compiler package identity is `reuse_apple`, matching
the existing package so package-only helpers remain accessible without changing
frozen source. The static framework shares one linked MLX runtime with the
candidate in the app.

Stage the verified streaming checkpoint as `BenchAssets/realtime-reuse.safetensors`
and a finite 16 kHz mono little-endian float32 fixture as `BenchAssets/input.f32`.
Model weights and audio stay outside Git. The default configuration runs a short
four-second comparison, then 32-second continuous repetitions, in balanced
baseline/candidate/candidate/baseline order. Configure only the short phase while
diagnosing correctness. State stays continuous across repetition seams within a
trial. Each trial starts with a fresh reset and three dummy warmup frames; loading,
warmup, output serialization and validation are outside the timed replay.

`paced: false` measures file replay as fast as possible. Optional `paced: true`
waits for synthetic 20 ms input arrivals and reports their processing lateness.
Neither mode measures microphone, acoustic echo cancellation, transport, codec,
jitter buffer or speaker latency.

Every trial saves raw frame times, wall time, thermal observations, finite sample
counts, an output SHA256 and the full output. Candidate output is compared to the
first baseline in the same phase, with exact float-bit equality reported separately
from the configured absolute tolerance. A parity failure saves partial evidence
and stops subsequent trials; it does not silently accept a new output.

Generate and build on the development Mac:

```sh
xcodegen generate --spec Benchmarks/StreamingEight/project.yml
xcodebuild -project Benchmarks/StreamingEight/ReuseStreamingEightBenchmark.xcodeproj \
  -scheme ReuseStreamingEightBenchmark -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath /tmp/Reuse-Eight-Derived \
  DEVELOPMENT_TEAM=YOUR_TEAM build
```

Replace `YOUR_TEAM` with the Apple development team authorized for the test device.
The project does not commit a developer's signing team.

Install and launch only on the intended unlocked physical test device. Bundle ID:
`io.riftco.reuse-streaming-eight-benchmark`. Launch with `--autorun`; the foreground
app disables auto-lock during the run and restores it afterward. Console events
begin `REUSE_EIGHT_BENCHMARK_EVENT=`. Results are atomically saved to
`Documents/latest-progress.json` after each trial and `Documents/latest-results.json`
after completion. Exceptions additionally produce `latest-failure.json`.

The first model load in a process may use system shader caches. Report it as a
load measurement, not a guaranteed uncached compile. Compare like-for-like phase,
trial order and thermal conditions; p95/p99 are nearest-rank order statistics.

After copying the completed results JSON from the device, produce a validated
summary with Python's standard library:

```sh
python3 Benchmarks/StreamingEight/summarize.py /path/to/latest-results.json > summary.json
```

The summarizer requires all eight trials, the pinned baseline and checkpoint,
eight-layer FP32 metadata, finite raw timings, and valid parity records. It pools
all frames separately for each phase and variant, retaining individual trial
means, thermal observations, and parity. The 32-second phase is primary; do not
combine it with the short phase. Mean latency savings compare equal frame counts;
wall-time RTF also includes the outer replay loop. No frames or outliers are
discarded. The fraction over 20 ms is a slow-frame fraction for file replay,
not a measured live-call deadline-miss rate. Two trials per variant do not support
treating individual frames as independent experimental repetitions.
