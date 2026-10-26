# RE-USE Audio for Android

Offline restoration using the commercial NVIDIA NemotronLabs RE-USE **30-layer, 9,607,747-parameter model in FP32**, executed by ONNX Runtime's CPU provider. This module does not contain the separately licensed `ReuseStreaming` model or its weights. Existing Swift products and APIs remain independent.

The Android backend preserves the model architecture and checkpoint. Float32 FFT, ONNX operator, and platform decoding/resampling implementations introduce numerical rounding differences; this is not a claim of bit-for-bit Apple output or real-time phone performance.

## Use from an Android application

The standalone build requires JDK 21 and Android SDK 36. Kotlin 2.2.20, Gradle 8.13 (SHA-pinned wrapper), AGP 8.11.1, ONNX Runtime 1.23.2, NDK 27.2.12479018 and CMake 3.22.1 are pinned. AGP can install the pinned native toolchain when SDK licenses are accepted. Host tests also require a macOS/Linux C++17 compiler. The library supports Android API 26+ and Java bytecode 17.

The published Android SDK is available from Maven Central:

```kotlin
// settings.gradle.kts repositories: google(); mavenCentral()
// app/build.gradle.kts
dependencies { implementation("io.github.quangng2000:reuse-audio-android:0.1.1") }
```

ONNX Runtime is resolved transitively. Provision the separately downloaded model
with the trusted manifest described below.

For development, you can instead include a Git-pinned checkout/submodule:

```kotlin
// settings.gradle.kts
includeBuild("dependencies/reuse-audio/android") {
    dependencySubstitution {
        substitute(module("io.github.quangng2000:reuse-audio-android"))
            .using(project(":reuse-audio"))
    }
}
// app/build.gradle.kts
dependencies { implementation("io.github.quangng2000:reuse-audio-android:0.1.1") }
```

This composite build works directly from source. Maven Central release preparation
and signing instructions are in [PUBLISHING.md](PUBLISHING.md). Root project:
`ReuseAudioAndroid`; library project: `:reuse-audio`.

```kotlin
import io.riftco.reuseaudio.*

// Run construction and enhancement on one dedicated background worker.
val manifest = ModelManifest.fromJson(trustedBundledManifest)
val cancellation = CancellationToken()
ReuseEnhancer(modelFile, manifest, InferenceOptions(intraOpThreads = 2)).use { enhancer ->
    val result = enhancer.enhanceFile(originalFile, temporaryWav, privateScratchDirectory,
        cancellation = cancellation, progress = { fraction -> /* finite, monotonic 0...1 */ })
    // Publish temporaryWav in your recording library only after successful completion.
}
// From another thread: cancellation.cancel(); wait for the worker to finish before reusing resources.
```

`enhance(FloatArray, cancellation, progress)` accepts finite, nonempty **16 kHz mono Float32 PCM** and returns the same number of samples. This raw PCM API matches Apple `ReuseEngine` windowing and does not normalize the output peak.

`enhanceFile(input, output, scratchDirectory, cancellation, progress)` accepts mono/stereo PCM16/24/32 or Float32 WAV and audio supported by the device's native MediaCodec decoder, including AAC/M4A. Import is bounded, downmixed, and normalized to 16 kHz with a windowed-sinc resampler. The result is a mono Float32 WAV with one whole-recording gain `min(1, .98 / peak)`, matching the app's peak policy. `EnhancementResult` reports `sampleCount`, `sampleRate`, `peakBeforeGain`, `appliedGain`, and `modelId`.

Input and output must differ. Inputs are never overwritten. Owned scratch files and unpublished output are removed on failure/cancellation; an existing output is replaced only at the final publish point. The file API uses disk-backed PCM and a bounded 64,000-sample window with an 8,000-sample overlap. WAV output must fit the RIFF 32-bit size limit. Temporary disk space is required for decoded, normalized, and provisional audio.

One enhancer serializes its jobs. Cancellation is permanent per token, is checked during import/DSP/windowing, and signals `OrtSession.RunOptions.setTerminate(true)` during native inference. Callers must wait for actual worker completion before releasing their processing lease. The SDK has no microphone, playback, user-library, billing, or network behavior.

## Model integrity and reproducible export

Models are not embedded in Git or the AAR. Applications provision a pinned ONNX artifact themselves. Bundle the **trusted manifest** with the application; do not accept an untrusted downloaded sidecar as the source of the expected hash. `ModelManifest.fromJson`/`read(File)` enforce the 30-layer source contract, and session construction verifies ONNX size and SHA-256 before loading.

The source checkpoint is `06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1`, architecture revision `31f82f2efad1095034492f236e8dac9239ba713b`. Shared metadata, licenses and local checkpoints live in [models](../models/). Export tooling and its independently implemented PyTorch CPU oracle live in [tools/model-export](../tools/model-export/). Only FP32 CPU inference is registered: no NNAPI, FP16, quantization, model truncation, or streaming approximation is enabled.

The graph consumes `magnitude`/`phase` and returns `enhanced_magnitude`/`enhanced_phase`, all `[1,321,frames]` Float32 with dynamic final-window lengths. The DSP uses periodic Hann, FFT 640/hop 80, centered reflect padding, log1p magnitude, the original sweep suppression, inverse overlap/add, and the same 64k/8k crossfade starts/weights. PocketFFT's pinned Float32 implementation and exact reference Hann coefficients preserve phase branches at the first reflected frame; their source hashes are recorded beside the C++/Kotlin source. JNI and host tests compile with `-O2 -fno-fast-math -ffp-contract=on`: ordinary FP32 fused operations match the reference, with no fast-math reassociation or reduced precision. Final short windows retain their actual padded length; padding every window to 64k would change instance normalization and bidirectional context.

## Build and validate

From `android/`, with `JAVA_HOME` pointing to JDK 21 and `ANDROID_HOME` to the Android SDK:

```sh
./gradlew :reuse-audio:testDebugUnitTest :reuse-audio:assembleRelease --max-workers=2
```

DSP/file tests cover independent FFT/STFT comparisons, sweep suppression, boundary lengths and clips beyond 60 seconds, bounded reads, whole-file gain, stereo conversion, cancellation, integrity failures, and destination preservation. Real model tests are explicitly skipped unless the generated oracle assets are supplied:

```sh
REUSE_PARITY_DIR="$PWD/../tools/model-export/generated/parity" \
REUSE_MODEL_FILE="$PWD/../tools/model-export/generated/reuse-30-fp32.onnx" \
REUSE_MODEL_MANIFEST="$PWD/../tools/model-export/generated/reuse-30-fp32.manifest.json" \
./gradlew :reuse-audio:testDebugUnitTest --max-workers=2
```

For Android instrumentation, stage the same generated ONNX/manifest and `640-*.f32` oracle files in `reuse-audio/build/generated/parity-assets/` before running `:reuse-audio:connectedDebugAndroidTest`. The generated assets remain ignored. Instrumentation checks actual Android CPU/JNI inference, PCM restoration, native AAC decode/resampling, and cancellation. An emulator provides functional evidence; physical-device latency, thermals, and battery behavior require a real phone.

`validation.json` describes the recorded validation build. Its artifact/source
hashes and evidence paths retain their historical identity; current reproduction
commands use the layout above.

## Attribution

The AAR includes the exact offline model license, NVIDIA/Mamba/port third-party notices, PocketFFT's BSD-3-Clause notice, and ONNX Runtime 1.23.2 license/notices under `assets/reuse-audio/licenses/`. Expose applicable notices in the consuming app. ONNX Runtime Java API and termination behavior are documented by [Microsoft](https://onnxruntime.ai/docs/get-started/with-java) and [RunOptions](https://onnxruntime.ai/docs/api/java/ai/onnxruntime/OrtSession.RunOptions.html). No noncommercial streaming assets are part of this module.
