# Apple bounded-window integration

The existing `ReuseAudio` product, `ReuseEngine(modelURL:optimization:)` initializer
and complete-array `enhance(_:progress:)` API remain available unchanged.

A file client can process one independent window without allocating a complete
recording in memory:

```swift
import ReuseAudio

let engine = try ReuseEngine(modelURL: checkpoint, optimization: .standard)
let paddedOutput = try engine.enhanceWindow(samples) { completedBlock in
    // Reports 0...30. Throwing here cancels before later blocks run.
}
```

Input is finite 16 kHz mono Float32, from 1 through 64,000 samples. The result
keeps the established zero padding: at least 640 samples, rounded up to a multiple
of 80. The input-length prefix is restored audio; the extra samples are padding.
The method checks task cancellation before inference, between blocks and before
returning. Reuse one engine from one serial worker.

This API performs the existing offline STFT, all 30 model layers and synthesis.
It does not add streaming state. A bounded file adapter owns its 8,000-sample
crossfade, final trimming, global output gain and durable file commit. The app
selects `.standard` explicitly to retain its original FP32 equations; experimental
optimization remains an opt-in through the existing API.

## Regression evidence

[Six real-checkpoint checks](apple-window-validation.json) passed using the pinned
commercial weights. Four public API tests cover padding, finite/bounded inputs,
all blocks, callback/task cancellation and reuse after cancellation. Two app
integration tests require bitwise equality against its retained vendored FP32
engine on short padded windows and over a 72,000-sample crossfade/file pipeline.
The original input file's bytes were preserved. Engine and test builds emitted
no diagnostics. These checks do not establish audio quality or device throughput.

The public contract tests are in
[`OfflinePublicWindowTests.swift`](../Tests/ReuseAudioTests/OfflinePublicWindowTests.swift).
The app repository retains the separate vendored-reference integration runner at
`iosApp/Tests/SDKIntegration/run.py`, with setup instructions alongside it. The
summary records the executed command and source hashes; the runner saves exact
compiler commands/logs and immutable source copies in a new output directory.
Generated models, objects and binaries are excluded from this evidence document.

## Separate PyTorch transform observation

A [same-PCM diagnostic](apple-pytorch-transform-observation.json) also compared the
unchanged Apple `.standard` path with the supplied PyTorch oracle. On the provided
640-sample fixture, Apple/PyTorch maximum waveform error was about 0.000971 and
relative RMS error was 4.26%. On the 64,000-sample speech fixture, those values
were about 0.001046 and 0.440%. Outputs had the exact expected lengths and all
samples were finite. No resampling or peak gain was applied.

Input-spectrum diagnostics showed small Hann/magnitude differences and raw phase
wraps near ±π, including 79 first-frame wrap differences on each fixture. This
records an existing frontend difference; it does not isolate every contributor
to waveform error or establish audio quality. The JSON's Android values describe
the earlier pre-PocketFFT diagnostic, not the final Android implementation.
No Apple math, output or tolerance changed. The Apple integration acceptance gate
remains bitwise equality with the retained Apple app path, as tested above.
