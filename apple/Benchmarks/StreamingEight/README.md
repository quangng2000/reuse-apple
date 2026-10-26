# Eight-layer FP32 device comparison

Layout note: runnable commands use the current `apple/` hierarchy. Dated reports
and source-hash manifests retain the paths from their measured revisions.

## Latest iPhone result and selected option, October 25, 2026

The selected path is **compiled decoder features plus pooled frequency
filters/scans**, retaining the original per-branch GEMMs and eager final phase
heads. It is enabled automatically for `StreamingEnhancer(modelURL: url, layers: 8,
optimized: true)`. Other exits and `optimized: false` keep their prior selection.

On the physical iPhone 13 Pro Max (A15, iOS 26.2.1), the full continuous ABBA
comparison against the previous optimized revision `2446c83` measured mean frame
time **28.610 → 27.229 ms (4.82% reduction)** and P95 **29.240 → 27.910 ms**.
There were two 32-second trials and 3,202 completed frames per variant. Candidate
wall RTF was 1.365: 32 seconds of audio took about 43.67 seconds to process.
Every frame exceeded the 20 ms hop, so **the sub-20 ms target was not reached**.

Three additional short ABBA screens tested compiled-only, pooling-only, and
body-only compilation plus pooling. The combined path had the best paired
candidate/baseline mean ratio. Body-plus-pooling was essentially tied in absolute
short mean (27.266 versus 27.281 ms); these short screens do not prove a global
optimum. No alternative met the predeclared threshold for a further full run.
Compiled-only had a retained 130.800 ms outlier. No samples or outliers were
removed from any comparison.

All **20 iPhone comparison trials** produced bit-for-bit identical FP32 output
to their phase baseline, with matching lengths and host-verified output SHA256s.
All thermal observations were nominal and low power mode was off. The input is
the same four-second diagnostic repeated continuously for the longer trials.
This measures completed frame processing in file replay, not end-to-end calls.

Evidence:

- [Full eight-trial results](Reports/2026-10-25-iphone13promax-iteration2-results.json)
  and [validated summary](Reports/2026-10-25-iphone13promax-iteration2-summary.json).
- [Individual-option screens](Reports/2026-10-25-iphone13promax-option-screen.json).
- [Measured engine source hashes](Reports/2026-10-25-measured-engine-10c9-source.json),
  matching engine commit `83a2a632d84215ce344cd74224836901efc8c285`.
- [iPhone stage profile](../Profiling/Reports/2026-10-25-iphone13promax-stage-profile.json).

The measured engine used explicit package selection of the same validated path.
Final public adoption changes only constructor selection and comments; per-frame
GPU math is unchanged. Public-path, legacy fallback, state, finish and reset tests
were rerun after adoption. `BenchAssets/candidate-source.json` describes the
current source for future runs; the archived measured hashes above preserve the
device measurements' original provenance.

## First iteration: projection sharing and frame completion

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
per-file hashes are in
[that iteration's source manifest](https://github.com/quangng2000/reuse-apple/blob/2446c8383ac65da5376d9159fa1d079535c4e36b/Benchmarks/StreamingEight/BenchAssets/candidate-source.json).
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

The current reference is pinned to `2446c8383ac65da5376d9159fa1d079535c4e36b`.
The candidate explicitly enables package-only `compilation: .decoderFeatures` and
`pooledFrequency: true`; the reference uses compilation disabled and no pooling.
Both native targets and the package use whole-module Swift compilation. The
application's compiler package identity is `reuse_apple` solely to access these
benchmark-only package options. Config and result JSON record both option sets.
Before a final build, replace `source-sha256:PENDING_SOURCE_FREEZE` in the staged
configuration with the independently verified frozen candidate source SHA256.
The app rejects the placeholder rather than labeling a new binary with an old hash.

The default remains the full four-second and 32-second ABBA plan. For bounded
ablation, append validated arguments when launching:

```text
--autorun --candidate-compilation body --candidate-pooling false --short-only
```

`--candidate-compilation` accepts `disabled`, `body`, or `decoderFeatures`;
`--candidate-pooling` accepts exactly `true` or `false`. Each option may appear
once. `--short-only` is a flag and runs only the four-second ABBA phase. Overrides
change neither the fixed eight-layer FP32 reference nor the model, input, or
candidate source hashes. Results and console events record actual effective
options, scope, planned phases, and the run UUID. A short ablation is a screening
measurement, not a replacement for sustained 32-second trials.

`paced: false` measures file replay as fast as possible. Optional `paced: true`
waits for synthetic 20 ms input arrivals and reports their processing lateness.
Neither mode measures microphone, acoustic echo cancellation, transport, codec,
jitter buffer or speaker latency.

Every trial saves raw frame times, wall time, thermal observations, finite sample
counts, an output SHA256 and the full output. Candidate output is compared to the
first baseline in the same phase. **Every float bit must match.** The configured
absolute tolerance and maximum absolute difference remain diagnostic information
and cannot accept differing bits. Validation occurs outside timed replay; the
first failed comparison saves partial evidence and prevents any subsequent trial.

Generate and build on the development Mac:

```sh
python3 apple/Benchmarks/Profiling/prepare.py
xcodegen generate --spec apple/Benchmarks/StreamingEight/project.yml
xcodebuild -project apple/Benchmarks/StreamingEight/ReuseStreamingEightBenchmark.xcodeproj \
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
Each new run has a UUID present in console events, reports, trial rows, failures,
and output filenames. Match the UUID from the current launch before accepting a
copied `latest-results.json`; an earlier successful file may still exist while a
new run is loading or after a failure.

The first model load in a process may use system shader caches. Report it as a
load measurement, not a guaranteed uncached compile. Compare like-for-like phase,
trial order and thermal conditions; p95/p99 are nearest-rank order statistics.

After copying the completed results JSON from the device, produce a validated
summary with Python's standard library:

```sh
python3 apple/Benchmarks/StreamingEight/summarize.py /path/to/latest-results.json \
  --run-id CURRENT_LAUNCH_UUID > summary.json
```

The summarizer requires all eight trials, a trusted pinned baseline and checkpoint,
eight-layer FP32 metadata, finite raw timings, and valid parity records. It pools
all frames separately for each phase and variant, retaining individual trial
means, thermal observations, and parity. The 32-second phase is primary; do not
combine it with the short phase. Mean latency savings compare equal frame counts;
wall-time RTF also includes the outer replay loop. No frames or outliers are
discarded. The fraction over 20 ms is a slow-frame fraction for file replay,
not a measured live-call deadline-miss rate. Two trials per variant do not support
treating individual frames as independent experimental repetitions.
Historical `bdf56d19f4dbc16fb9ee949d42b70b4b6b5fe43e` reports remain supported without
`--run-id`; they are labeled as legacy reports without a UUID. Current `2446c83`
comparisons additionally require a valid UUID, recorded engine options, matched
whole-module settings, and bitwise equality for all completed trials.
An explicitly declared current short-only ablation is also accepted with exactly
four ABBA trials and matching metadata; its summary identifies the limited scope.

For a separate pinned stage diagnostic, launch with `--profile-only`; see
[`../Profiling/README.md`](../Profiling/README.md). Its forced completion boundaries
distort normal execution and its timings must not replace the uninstrumented
comparison.
