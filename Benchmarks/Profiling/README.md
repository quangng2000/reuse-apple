# Eight-layer stage diagnostic

## Current candidate and validation

The next candidate pools the two independent frequency filters/scans while
keeping their six original GEMMs, then reuses a compiled encoder/TF/decoder-feature
graph. The final phase-head convolutions and atan2 remain eager: compiling those
heads changed output bits and was rejected. All model caches are explicit graph
inputs/outputs, initialized by an eager first frame and preserved correctly across
warmup, finish and reset. Pooling is configured before warmup and must stay fixed
for the compiled graph's lifetime.

On an M4 Pro Mac, an uncontended Release build without coverage counters measured
5.697 ms baseline versus 4.589 ms candidate mean frame processing time (19.45%
reduction). P95 was 6.610 ms versus 5.365 ms. Both variants processed two 32-second
trials in ABBA order, retaining all 3,202 frame timings per variant. Every output
sample was bitwise equal, all thermal samples were nominal, and low power mode
was off. These are Mac results; the candidate has **not yet been benchmarked on
the iPhone**, and no sub-20 ms phone result is established. Public initializer
defaults remain unchanged until device validation.

The complete correctness suite passed 26 tests. Three optional benchmark entry
points were skipped in that suite and exercised separately. Kernel rewrites that
precomputed frequency exponentials or distributed temporal state across SIMD
lanes were exact but slower on Mac, so they were removed. Batched GEMMs changed
float bits and were rejected. Pooled-scan threadgroups 32/64/128 differed by less
than 0.3% in a short Mac check, so the existing 128-thread geometry is retained.

Evidence:

- [Combined candidate raw ABBA results](Reports/2026-10-25-m4pro-combined-candidate.json).
- [Clean stage diagnostic](Reports/2026-10-25-m4pro-stage-profile.json).
- [Rejected scan experiments](rejected-scan-experiments.json).

The stage diagnostic attributed about 70% of instrumented wall time to TF0–TF7
combined, with decoding the largest individual stage. Its added fences increased
mean time from 5.468 ms to 9.013 ms; use it for attribution, not throughput claims.

## Diagnostic implementation

This target attributes work in the already measured eight-layer FP32 implementation
at commit `2446c8383ac65da5376d9159fa1d079535c4e36b`. It is separate from shipping
package code and the normal baseline/candidate comparison.

Prepare the ignored source snapshot before generating the benchmark project:

```sh
python3 Benchmarks/Profiling/prepare.py
xcodegen generate --spec Benchmarks/StreamingEight/project.yml
```

The generator obtains each source with `git show` at the pinned commit, regardless
of edits to the working tree. It applies exact-match hooks to the generated copy,
records original/generated file hashes plus generator/support hashes, and fails
if a hook no longer matches. The private static Xcode framework uses the original
`reuse_apple` package identity for access to existing package helpers and shares
the application's linked MLX runtime. Generated sources are excluded from Git.
Native reference, diagnostic, and application targets explicitly use whole-module
Swift compilation, matching the local Swift package's compilation mode.

After the usual signed build and installation, launch the benchmark app with
`--profile-only`. This runs automatically without a microphone or network input.
It uses the same verified checkpoint and the first 15,040 samples of the staged
fixture. The normal and instrumented runs each warm three dummy frames and reset,
then process eight unmeasured real frames with continuous recurrent state before
40 measured frames, including EOF. Dummy frame/kernel warmup and final cache
readback/hashing are outside replay timing.

The diagnostic forces evaluation and stream completion after analysis, encoding,
each of eight time-frequency blocks, decoding, and synthesis. Every model boundary
also evaluates all updated convolution and recurrent caches. Each stage records:

- Host time spent constructing the lazy graph.
- Time in `eval` plus `Stream.gpu.synchronize()`, including host traversal and
  command encoding, GPU work and completion wait.
- Synthesis readback and CPU overlap-add time after completion.

**Evaluation wait is GPU-completed wall time, not isolated GPU execution time.**
The added boundaries change dispatch, fusion and overlap. These measurements
identify broad areas to inspect; the ordinary uninstrumented ABBA comparison
determines whether an optimization is faster. A Metal System Trace can provide
actual GPU command/kernel attribution separately. MLX's `GPU.startCapture` also
requires an MLX build with `MLX_METAL_DEBUG` and `MTL_CAPTURE_ENABLED=1`; this
diagnostic does not enable it or alter MLX build flags.

The hooks-off reference and stage-instrumented run use the same pinned diagnostic
module. The app compares every output's float bits and SHA256s over every full
FP32 convolution/recurrent cache and transform overlap/envelope array. A mismatch
is saved and reported; no output tolerance is substituted for exact equality.
The final cache gate complements the full output comparison and does not claim
to inspect every intermediate frame's cache.

The app writes `Documents/latest-profile.json`, the two `.f32` outputs, per-frame
stage timings, stage aggregates, normal frame timing, source manifest, thermal
observations, and output/state gate evidence. Console events begin
`REUSE_EIGHT_PROFILE_EVENT=`. Errors additionally write
`Documents/latest-profile-failure.json`.

The same Foundation-only runner can execute on a development Mac while the phone
is unavailable. This identifies host-machine bottlenecks and does not substitute
for A15 timings. Use a separate derived directory and coordinate exclusive GPU
access with other performance work:

```sh
xcodebuild -project Benchmarks/StreamingEight/ReuseStreamingEightBenchmark.xcodeproj \
  -scheme ReuseStreamingStageProfileMac -configuration Release -destination 'platform=macOS' \
  -derivedDataPath /tmp/ReuseEightStageMacDerived build
/tmp/ReuseEightStageMacDerived/Build/Products/Release/ReuseStreamingStageProfileMac \
  Benchmarks/StreamingEight/BenchAssets \
  Benchmarks/Profiling/Generated/profile-source.json /tmp/reuse-eight-mac-stage-profile
```
