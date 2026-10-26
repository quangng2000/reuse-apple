# Apple SDK

The Apple runtime lives here. The repository-root `Package.swift` remains the
Swift Package entry point, with the same products and imports:

```swift
import ReuseAudio
let engine = try ReuseEngine(modelURL: checkpointURL)
let restored = try engine.enhance(mono16kSamples)
```

- `Sources/ReuseAudio/` — the 30-layer FP32 offline engine for iOS and macOS.
- `Sources/ReuseStreaming/` — the separately licensed Apple streaming research engine.
- `Tests/` — Swift package tests for both products.
- `Tools/ReuseProbe/` — offline command-line executable source.
- `Examples/` — runnable Xcode projects, streaming CLI and integration checks.
- `Apps/OfflinePrototype/` — UI source used by the offline example project.
- `Benchmarks/` — profiling, device benchmark apps and frozen reference sources.
- `Research/` — streaming checkpoint provenance and its separate license.
- `scripts/` — Apple build and smoke-test entry points.

Shared model provenance and licenses live in [`../models`](../models). The
portable offline checkpoint fetcher and ONNX conversion tools live in
[`../tools/model-export`](../tools/model-export). Android's runtime is separate
under [`../android/reuse-audio`](../android/reuse-audio).

Run commands from the repository root. For example:

```sh
python3 tools/model-export/prepare_model.py
bash apple/scripts/run_mac.sh --input noisy.wav --output enhanced.wav
```

See [package integration](../docs/package.md), [development](../docs/development.md),
and [licensing boundaries](../NOTICE). The public Swift names and model behavior
are unchanged by the directory move.
