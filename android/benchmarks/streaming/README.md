# Android streaming benchmark

By default this standalone release app consumes the published
`io.github.quangng2000:reuse-streaming-android:0.1.3` artifact. It measures FP32
CPU streaming with pinned research models and synthetic input. Explicit local
build flags also allow testing the direct-buffer optimization and opt-in native
frequency kernel. The native variant preserves all weights, layers and states
but has small FP32 rounding differences; it is not bitwise identical to the
published graph.

The original emulator experiment below depends on the host CPU, hypervisor,
guest configuration and concurrent host load. Its timings do not establish Android
phone performance, sustained thermal behavior, battery cost or full call latency.
[Android recommends physical devices for representative benchmarks](https://developer.android.com/topic/performance/benchmarking/benchmarking-in-ci).

## Current Vulkan selection and archived comparisons

The local Vulkan SDK now accepts **only the audited v2 bundle**, with manifest
SHA-256 `951adf47ebece2c27a5af4df22b91c30c8a442b8caa04bb01db102ceb0ac733a`.
`ExperimentalVulkanStreaming.create()` selects its measured tiled kernels;
`createTiled()` is an alias. v1 and v3 manifests are rejected. This does not
change the historical 0.1.2 CPU measurements. New benchmark builds report the
0.1.3 coordinate; the ordinary constructor still uses CPU inference.

The last same-phone ABBA comparison measured **17.69 ms mean for v2** and
**18.55 ms for v3**: v3 was 4.87% slower. Both pooled p95 values were 19.59 ms,
but each backend had 4/320 calls above 20 ms. These are historical short-run
Galaxy S25 Ultra results, not new measurements or sustained real-time claims.
See [the public v3 comparison](results/firebase-vulkan-v3-galaxy-s25-ultra-2026-10-26.json)
and [the Vulkan guide](../../../tools/android-streaming-vulkan/README.md).

The old direct/tiled and multi-bundle Kotlin instrumentation is retained under
[`archive/vulkan/`](archive/vulkan/) and excluded from the active instrumentation
source sets. The result JSONs and historical verification tools remain available. The active
`VulkanProfileTest.kt` is a v2-only diagnostic; its instrumented timings are not
normal latency measurements. To
reproduce those older comparisons, use a separate checkout of
[commit ad4ceda](https://github.com/quangng2000/reuse-audio/tree/ad4ceda) with its
matching SDK, exact bundles and fixtures. The current v2-only SDK intentionally
cannot execute the old v1/v3 cases. Do not re-enable archived cases in the
active benchmark or relax artifact admission for a historical comparison.

For future Vulkan integrations use `create()` with the audited v2 bundle.
`createProfiled(..., tiled = true)` is available for diagnostic timestamps only;
exclude profiled calls from normal latency comparisons. No additional physical
test was performed for this selection update; a package release does not create
new performance evidence. Generated
model bundles, native binaries and personal recordings remain outside Git.

## Recorded emulator result

The [26 October 2026 measurement receipt](results/emulator-2026-10-26.json)
contains all raw timings and verification evidence. This ARM64 Android 16/API 36
emulator used four virtual cores and 4 GiB RAM on an Apple M4 Pro host.

Eight layers with four inference threads was the fastest tested eight-layer
configuration. The separate 1,000-hop confirmation averaged 88.39 ms per 20 ms
audio hop (median 87.64 ms, p95 93.34 ms, p99 105.80 ms; RTF 4.42).
All 1,000 hops exceeded 20 ms. Process PSS was 293.94 MiB at the after-finish
boundary snapshot, not a peak measurement. Reference waveform checks, bitwise
reset repeatability, finite output and exact output length passed.

This CPU implementation did not sustain real time in the measured emulator.
The result does not predict an Android phone's speed. Host load was uncontrolled;
the two 300-hop eight-layer/two-thread trial means differed by about 1.1%.

## Build and run

Requirements match the SDK: JDK 21, Android SDK 36, and the SHA-pinned Gradle
wrapper in `android/`. From the repository root, first download and verify the
public model fixtures if they are not already exported locally:

```sh
python3 android/tools/fetch_streaming_assets.py --output /tmp/reuse-streaming-benchmark-assets
python3 android/benchmarks/streaming/tools/prepare_assets.py \
  --models /tmp/reuse-streaming-benchmark-assets/models \
  --parity /tmp/reuse-streaming-benchmark-assets/parity
```

The staging script verifies every model, trusted manifest, reference receipt and
raw fixture against checked-in pins. Generated APK assets stay under the ignored
`app/build/benchmark-assets`. Models and personal recordings must not be committed.

```sh
./android/gradlew -p android/benchmarks/streaming --no-daemon --max-workers=2 :app:assembleRelease
emulator -avd ReuseAudioSdkTest -port 5554 -cores 4 -memory 4096 \
  -no-window -no-audio -no-boot-anim -no-snapshot-load -no-snapshot-save -gpu auto
adb -s emulator-5554 wait-for-device
# Wait until: adb -s emulator-5554 shell getprop sys.boot_completed -> 1
adb -s emulator-5554 install -r android/benchmarks/streaming/app/build/outputs/apk/release/app-release.apk
python3 android/benchmarks/streaming/tools/run_benchmark.py --frames 300 \
  --output Validation/AndroidEmulatorStreaming-0.1.3/sweep-300
```

Use the SDK's absolute `emulator`/`adb` paths or add their directories to PATH.
Set `JAVA_HOME` and `ANDROID_HOME` for the local installation. The app is
nondebuggable, has no network/microphone permission, and uses a local development
signing key only for installation. Its runtime result records the actual
debuggable flag and rejects a debuggable target. The direct ADB emulator runner
does not use an instrumentation process or a benchmark-framework emulator
override. The optional Firebase runner is described separately below.

The default sweep is 8 layers / 2 threads, 4/2, 8/1, 8/4, 12/2, and 8/2 again.
For a longer selected configuration:

```sh
python3 android/benchmarks/streaming/tools/run_benchmark.py \
  --layers 8 --threads 4 --frames 1000 \
  --output Validation/AndroidEmulatorStreaming-0.1.3/eight-four-threads-1000
```

The command requires the named emulator, captures a unique run ID and validates
the chunked logcat result's SHA-256. `results.json` preserves raw per-hop timings,
while `environment.json` records host load/CPU/RAM and the guest fingerprint,
ABI, core count and RAM. Use a fresh output directory for every run. It saves
partial failed results and exits nonzero on failure. Stop only the owned emulator
after testing with `adb -s emulator-5554 emu kill`.

## What the measurements mean

- Before timing each case, a pinned independent NVIDIA CPU waveform reference
  must pass the existing 0.0003 absolute error bound. Four/eight layers use
  16,000 samples; twelve layers uses its available 640-sample reference. Reset
  is checked for bitwise repeatability.
- Each stream calls `warmUp()`, starts with 640 samples, and processes 32 more
  untimed hops before measurement. The initial 640-sample call renders two
  internal frames; steady calls each accept 320 samples and render one frame.
- Primary latency is monotonic wall time around the entire public `push()` call.
  Input generation, output validation, model provisioning and memory sampling
  occur outside that timer. Internal render timing is a secondary measurement,
  not pure neural-network inference time.
- Percentiles use nearest rank. The 20 ms comparison is the processing budget
  for one 320-sample hop at 16 kHz. Processing is unpaced; no sleep simulates
  audio capture. RTF is the sum of measured push times divided by timed audio
  duration; values above 1 cannot keep up in that measured environment.
- Construction, warmup, reset-stream startup and EOF flushing are separate.
  Startup is measured with a warm graph after reference validation; it is not
  cold application or cold-storage startup. Capturing the initial 640 samples
  itself takes 40 ms, which the processing timer excludes.
- PSS/Java/native memory are boundary snapshots, not peak measurements. Raw
  Android GC counters are retained; reported allocation/GC values are approximate.
  See [Android Debug runtime statistics](https://developer.android.com/reference/android/os/Debug#getRuntimeStat(java.lang.String)).
- Every run checks finite output and exact total sample count after `finish()`.
  Synthetic reference parity is a functional numerical check, not a perceptual
  audio-quality test.

The streaming model and derivative SDK are NSCLv1, noncommercial research and
education only. Required licenses/notices are packaged with the generated
benchmark APK. A separate commercial license is required for paid-app use.

## Local eight-layer optimization candidate

The [optimization receipt](results/optimization-2026-10-26.json) preserves raw
timings, host/guest metadata, archived APK hashes and numerical checks. With the
same four-core emulator, published baseline medians before and after the
experiment were 88.48 and 86.00 ms. The direct-buffer median was 62.44 ms.
The experimental native frequency kernel at four threads confirmed a 23.82 ms
median across 1,000 hops (mean 23.93 ms, p95 25.46 ms). All 1,000 hops exceeded
20 ms. This is approximately 3.6–3.7 times faster than the baseline brackets,
but does not meet the 20 ms target in this emulator. These timings do not predict
phone performance; the physical-phone result is recorded below. Concurrent host
load and CPU frequency were not controlled.

Standard-graph direct-buffer host tests remained bitwise identical to a frozen
copying reference. The native kernel passed full state, reset and waveform
checks with numerical rounding differences. On one exact four-second synthetic
noisy-speech sample, candidate versus published output had maximum absolute
error 3.43e-7, relative RMS error 4.40e-7 and agreement SNR 127.13 dB. STOI changed
by -2.76e-9 and SI-SDR by -3.97e-7 dB. This is one diagnostic sample, not broad
audio-quality or listening validation. The receipt includes exact PCM hashes.

For standard-graph local testing use `-PstreamingSdkSource=local` and a descriptive
`-PstreamingVariant=direct-buffers` when building. The native candidate additionally
requires the validated graph/library from
[`tools/android-streaming-native`](../../../tools/android-streaming-native/README.md),
staged with `python3 android/benchmarks/streaming/tools/prepare_native_assets.py`,
and build flags `-PstreamingVariant=native-frequency-v1 -PstreamingNativeScan=true`.
Always select `--layers 8` for this experimental variant. Model and library bytes
are SHA checked before registration; generated assets stay out of Git.

## Firebase physical phones

The [Firebase Test Lab runner](tools/firebase/README.md) launches the same release
app through AndroidJUnitRunner and measures CPU work inside the phone. Both
published and native app/test APK pairs built and passed local smoke tests, with
SHA-verified raw artifact extraction. The runner rejects emulators by default;
the smoke tests explicitly allowed the owned emulator. These smoke tests are
not cloud-phone measurements. Firebase project selection and fresh Google
authentication are required before checking device availability/quota. No cloud
test, APK upload or billing change was performed during preparation.

### Verified physical Galaxy S25 Ultra result

Both APK pairs passed on physical Galaxy S25 Ultra
devices (`pa3q`, SM-S938U1, Android 15/API 35). The
[physical measurement receipt](results/firebase-galaxy-s25-ultra-2026-10-26.json)
preserves all 300 raw push timings per build, verified artifacts and measurement
settings. Account, billing and local-path metadata are omitted from the public
results. Video and Firebase performance monitoring were off.

Published 0.1.2 averaged 181.72 ms per update (median 205.06 ms, p95 231.19 ms).
The optimized native FP32 candidate averaged 49.73 ms (median 46.44 ms,
p95 73.87 ms): a 3.65x mean speedup. All 300 measured updates in each build
exceeded 20 ms; the optimized RTF was 2.49. The physical device also missed the
target, so the latency issue cannot be attributed solely to the emulator.

Both XML reports contain one passing test and no failures/errors/skips. Uploaded
APK hashes, exact model/input identities, result hashes and sample counts were
independently verified. Candidate waveform reference maximum error was 2.71e-9;
reset replay remained bitwise repeatable. The two tests used the same phone
model/OS/fingerprint, but are not guaranteed to use the same handset. Clock and
thermal state were uncontrolled; this six-second unpaced input experiment is
not a sustained-call or all-phone guarantee.

Provider test durations were 69 and 19 seconds, attempt 1 each. Both executions
finished. These timings describe the initial comparison; subsequent experiments
are recorded separately below. The optimization remains experimental.

## Further FP32 native optimization and same-phone comparison

The [V2 physical receipt](results/firebase-galaxy-s25-ultra-v2-2026-10-26.json)
records a further operator profile and one ordered five-case comparison on a
physical Galaxy S25 Ultra/API 35. V1 profiling identified large temporal-state
arithmetic and scalar Softplus costs; finite checks were about 0.94 ms and were
retained. The new experimental backend fuses temporal state arithmetic, inlines
the exact FP32 MLAS exponential, and vectorizes stable Softplus with pinned Eigen
packet math and scalar extreme/nonfinite fallbacks. Default/V1 backends remain
available; all eight layers, weight bytes, 31 states and zero lookahead remain.

On the same handset the ordered mean push times were: V1/four threads 67.22 ms,
V2/four 48.63 ms, V2/eight 31.95 ms, V2/six 47.39 ms, and V1/four again 56.41 ms.
The fastest configuration had median 30.36 ms, p95 44.83 ms, RTF 1.60 and all
300 updates above 20 ms. The target was not reached. The 16.1% control drift means
the complete eight-thread gain cannot be attributed solely to kernel changes;
same-four-thread V2 was 13.8–27.6% below the bracketing controls. Clocks, thermals
and scheduler placement were uncontrolled. Prior 49.73 ms V1 measurements used
a different handset/firmware and are not the control for this execution.

All 63 SDK host tests passed, including carried-state/tail/reset/registry checks.
Final-library Exp matched 2,637,824 test values at 0 ULP. Softplus matched
2,000,008 values within four ULP while preserving tested subnormals and extreme
fallback behavior. On the four-second synthetic speech diagnostic, maximum
output difference from the published backend was 2.68e-7 and relative RMS error
4.33e-7. Predeclared numerical, STOI and SI-SDR gates passed; this is not a broad
listening study. Model/native-library hashes, raw results and one-pass physical
XML reports were independently checked. FP32 output is numerically equivalent,
not promised bitwise identical to the published graph.

Use `prepare_native_v2_assets.py` after its pinned proofs pass, then build the
local release benchmark with `-PstreamingVariant=native-v2-comparison`. The
`NativeComparisonBenchmarkTest` verifies the ordered pinned backends; the
independent `tools/firebase/extract_comparison.py` checks raw hashes and recomputes
all statistics. The stager includes the corresponding Eigen headers and license
notices in the APK. Profile/comparison test durations were 2 and 102 seconds.
Both executions finished without retries. Code remains experimental; publishing
these sources does not change the released SDK defaults.
