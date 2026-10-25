# Use the Swift Package

`Package.swift` exposes two separate library products. Select `ReuseAudio`
for the offline model, or `ReuseStreaming` for the noncommercial streaming model.
Importing the offline product does not include the streaming sources or weights.
See [NOTICE](../NOTICE) before choosing a product for commercial software.

## Swift Package Manager

Add this dependency to an Apple-platform package:

```swift
dependencies: [
    .package(url: "https://github.com/quangng2000/reuse-apple", from: "0.2.0")
],
targets: [
    .target(name: "YourApp", dependencies: [
        .product(name: "ReuseAudio", package: "reuse-apple")
    ])
]
```

Replace the product with `ReuseStreaming` when using its streaming API.
Xcode's File → Add Package Dependencies accepts the same repository URL.
Build for Apple silicon with Xcode 16.4+, macOS 14+ or iOS/iPadOS 17+.
MLX Swift is pinned to 0.29.1. Use Xcode/xcodebuild for the final build: plain
`swift build`/`swift run` cannot compile the required MLX Metal shaders.
Linux, Intel Macs and simulator performance are not supported or benchmarked.

## Offline audio

```swift
import ReuseAudio

let engine = try ReuseEngine(modelURL: checkpointURL)
let input = try AudioFileIO.read(inputURL)
let restored = try engine.enhance(input) { fraction in
    // Dispatch UI updates to the main actor, if needed.
}
try AudioFileIO.write(restored, to: outputURL)
```

`AudioFileIO` converts supported mono/stereo recordings to 16 kHz mono. The
offline engine accepts finite float samples and uses four-second windows with
half-second overlaps. This differs from the upstream full-recording model's
context. Run inference on a background worker and use one caller per engine.
The public API preserves FP32; it does not offer an FP16 quality/speed switch.

## Model files

The package does not automatically download or bundle model weights:

```sh
python3 Scripts/prepare_model.py
python3 Scripts/prepare_streaming_model.py
```

The offline model is saved to `Resources/model.safetensors`; the streaming
model to `Resources/realtime-reuse.safetensors`. Each loader checks a pinned
SHA-256 before inference. The streaming setup accepts `--output FILE` and saves
its license alongside that file. Pass the selected checkpoint's URL to the API.

For an app, bundle the correct model and license as app resources or implement
your own explicit download/storage flow. Keep the two checkpoints distinct.
See [streaming lifecycle](streaming.md) and [build checks](development.md).

The original offline iPhone app is in `Examples/OfflineApp/ReusePocket.xcodeproj`.
Both example apps use the root package as a local dependency.
