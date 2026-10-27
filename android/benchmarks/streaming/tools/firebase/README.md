# Physical Android streaming measurements with Firebase Test Lab

This adds a standard `AndroidJUnitRunner` test around the existing release benchmark app. The test launches one eight-layer case, sleeps while the app measures inference, verifies its final result, and copies artifacts after measurement. It adds no network client, microphone access, private recordings, tracing, or inference implementation.

Nothing here authenticates, enables a service, uploads an APK, or submits a cloud test automatically. The existing project, available physical-device quota, and maximum test executions must be resolved before a submission. Test Lab installation/test time and artifact storage can consume quota or incur charges; use the project's actual current limits.

## Build setup

The new source is `app/src/androidTest/kotlin/io/riftco/reusestreaming/benchmark/FirebaseStreamingBenchmarkTest.kt`. The benchmark app uses these Gradle settings while retaining a nondebuggable release target and matching test signing config:

```kotlin
android {
    defaultConfig {
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    testBuildType = "release"
}
dependencies {
    androidTestImplementation("androidx.test:runner:1.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
}
```

These are the stable versions listed in the official [AndroidX Test releases](https://developer.android.com/jetpack/androidx/releases/test). No Espresso/UI Automator polling, Orchestrator, coverage, or benchmark plugin is needed. MainActivity reports a device-sensitive `environmentScope` plus fingerprint, model, ABI, and `emulatorReported`.

Build `:app:assembleRelease` and `:app:assembleReleaseAndroidTest` for each APK variant only after the emulator timing window. Archive app APK, test APK, source revision/diff, source variant, and SHA256s together. Android test APK and app must come from the same variant/build and use matching signatures. Retain a published-SDK baseline app and a candidate app; do not silently replace the baseline with new code.

Before using quota, validate the runner locally on the owned emulator with `benchmarkRequirePhysical=false`; this only validates launch/result transport and is not a phone performance measurement. The runner defaults to rejecting emulators.

## Arguments and invariants

Arguments are strings passed using `gcloud --environment-variables`, which forwards them as instrumentation extras:

- `benchmarkRunId`: safe unique ID, at most 80 ASCII filename characters. Default is timestamped.
- `benchmarkFrames`: 32–1500 measured 320-sample hops; default 300.
- `benchmarkThreads`: 1–8 CPU threads; default 4. Exactly one layer-8 case executes.
- `benchmarkRequirePhysical`: default `true`; only set false for local runner validation.
- `benchmarkExpectedVariant`, `benchmarkExpectedSdkSource`: optional exact checks against APK build metadata. Set both for a controlled comparison.
- `benchmarkExpectedModelSha256`, `benchmarkExpectedInputSha256`: optional exact identity checks. Use model hash appropriate to the original graph or validated native graph.
- `benchmarkCaptureOutput`: default `false`, no waveform copies in performance runs.
- `benchmarkInputAsset`: optional packaged finite FP32LE mono 16 kHz validation asset. For the frozen 64,000-sample synthetic speech input, use `validation/noisy-speech.f32`, `benchmarkFrames=166`, `benchmarkCaptureOutput=true`, and expected input hash `c449b1ebf8f51c80f7530961b41c62a733a1033a60140fc16c0b37186120835f`. This is a separate numerical/audio quality run, not the performance comparison.

The test verifies nondebuggable ARM64/FP32/CPU operation, finite individual timings, exact output sample count, independent waveform fixture, reset repeatability, selected build/input/model identities, and a physical device when requested. A failed run remains a failure; no automatic retries are requested. Waiting is bounded to eleven minutes; MainActivity has its existing ten-minute bound.

## Submission template — review before executing

Read-only catalog discovery belongs to the root task: `gcloud firebase test android models list --project PROJECT_ID` and `gcloud firebase test android models describe MODEL_ID --project PROJECT_ID`. Select an available **physical** ARM64 phone/OS combination, not a virtual `*.arm` device. Do not use the illustrative placeholders below as actual IDs.

The following is an inert template, not a script run by this directory:

```sh
gcloud firebase test android run \
  --project PROJECT_ID \
  --type instrumentation \
  --app /absolute/archive/baseline/app-release.apk \
  --test /absolute/archive/baseline/app-release-androidTest.apk \
  --device model=PHYSICAL_MODEL_ID,version=SUPPORTED_API,locale=en,orientation=portrait \
  --test-targets 'class io.riftco.reusestreaming.benchmark.FirebaseStreamingBenchmarkTest' \
  --environment-variables benchmarkRunId=UNIQUE_PAIR-baseline,benchmarkFrames=300,benchmarkThreads=4,benchmarkExpectedVariant=published,benchmarkExpectedSdkSource=published,benchmarkRequirePhysical=true \
  --directories-to-pull /sdcard/Android/data/io.riftco.reusestreaming.benchmark/files/firebase-benchmark \
  --no-performance-metrics --no-record-video --no-use-orchestrator \
  --num-flaky-test-attempts 0 --timeout 12m \
  --results-dir UNIQUE_PAIR-baseline
```

For the candidate use its matching archived app/test APK pair, a new run/results ID, and its actual `benchmarkExpectedVariant`/`benchmarkExpectedSdkSource`; retain the same physical model, OS, threads, input, and frame count. Set exact expected model/input hashes once known. Each invocation schedules a test and may consume quota or money; never loop submissions to find a better latency.

Test Lab defaults performance monitoring and video on; both are disabled in this template so the existing in-app timers are the measurement source. Do not use cloud elapsed time, streamed screen response, network latency, startup/install time, or Firebase CPU graphs as per-hop inference latency. `rawPushMs` is the individual SDK push wall time, including runtime/state handling; setup, warmup, checks, EOF, and artifact transfer remain separately reported. Quality capture copies are intentionally excluded from performance runs.

Separate Test Lab executions of the same model/OS are not guaranteed to use the exact same physical handset. A baseline/candidate difference can include unit, thermal, scheduler, and frequency variation. Treat one pair as preliminary; any additional counterbalanced/repeated comparisons need an agreed quota. The custom harness does not lock CPU clocks or provide Jetpack Benchmark's thermal adaptation. Report raw distributions, p50/p95, over-20ms count, RTF, model/OS/build hashes, and this limitation; never treat a single result as an all-phone guarantee. Android's [benchmarking in CI guidance](https://developer.android.com/topic/performance/benchmarking/benchmarking-in-ci) recommends physical devices and evaluating noisy timing trends.

## Raw artifact verification

After the test completes, download its result tree from the exact Cloud Storage location printed by Test Lab. Do not guess a bucket or pull unrelated project files. The runner writes only its own raw JSON, a SHA256 receipt, and explicitly requested synthetic quality PCM to:

`/sdcard/Android/data/io.riftco.reusestreaming.benchmark/files/firebase-benchmark/`

Android scoped storage can limit pull access on some API levels. The existing `REUSE_BENCH` logcat emits the full result in indexed Base64 chunks plus SHA256, providing a raw-JSON fallback; PCM is not considered verified through that fallback. These are documented Test Lab [directory-pull and environment flags](https://docs.cloud.google.com/sdk/gcloud/reference/firebase/test/android/run), and [results live in Cloud Storage](https://firebase.google.com/docs/test-lab/android/analyzing-results).

Verify an already-downloaded artifact tree locally:

```sh
python3 tools/firebase/extract_results.py \
  --artifacts-root /absolute/downloaded/execution-tree \
  --output /absolute/new/verified-directory
```

Or, for JSON-only fallback:

```sh
python3 tools/firebase/extract_results.py \
  --logcat /absolute/downloaded/logcat \
  --output /absolute/new/verified-directory
```

The tool does not call gcloud, access the network, or alter downloaded files. It preserves exact result bytes, checks the receipt hash/size or logcat transport hash, verifies case invariants, checks requested PCM export hashes/sizes, and writes `verified-summary.json`. It rejects incomplete or failed app results. Keep failed raw downloads for diagnosis rather than representing them as timing evidence.

Run the lightweight extractor fault tests with:

```sh
python3 -m unittest discover -s tools/firebase -p 'test_*.py'
```

No cloud test or build was performed while preparing these files.
