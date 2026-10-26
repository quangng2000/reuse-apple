# RE-USE Audio

**On-device speech enhancement for Apple and Android.**
The Apple SDK uses Swift, MLX and Metal. The Android SDK uses Kotlin, CPU ONNX
Runtime and a native FP32 FFT. Both offline backends use the same pinned
30-layer NVIDIA NemotronLabs RE-USE checkpoint. Audio stays on the device;
inference needs no Python, CUDA, or server.

- **`ReuseAudio`** — offline enhancement with NVIDIA NemotronLabs RE-USE.
  9.61M parameters, FP32, OpenMDW-1.1 license; commercial use permitted.
- **`ReuseStreaming`** — incremental enhancement with NVIDIA Real-time RE-USE.
  Four-layer default, FP32, 16 kHz mono; **noncommercial research only**.
- **Android offline SDK** — all 30 layers in FP32, with WAV/AAC/M4A import,
  bounded file processing, progress and cancellation. See [Android setup](android/README.md).
- Mac command-line tools and iPhone/iPad examples consume the existing Swift APIs.

The Swift products, module names and existing public APIs remain unchanged.
`Package.swift` retains the `ReuseApple` package name for compatibility. The
additive [single-window API](docs/apple-bounded-window.md) lets apps keep bounded
file transport without copying the neural engine into application sources.

## Add the Apple package

In Xcode, add `https://github.com/quangng2000/reuse-audio` as a package dependency
and select the library product you need. Requirements: Apple silicon,
macOS 14+ or iOS/iPadOS 17+, and Xcode 16.4+.

```swift
import ReuseStreaming

let enhancer = try StreamingEnhancer(modelURL: checkpointURL)
try enhancer.warmUp() // Optional: prepare GPU kernels before capture.
let restored = try enhancer.push(mono16kSamples)
let finalSamples = try enhancer.finish()
```

Call a stream serially from a background worker. It preserves state across
chunks and emits audio when enough samples arrive. Download model weights
separately. See [Swift Package integration](docs/package.md) and
[streaming API and examples](docs/streaming.md).

## Try the tools

```sh
git clone https://github.com/quangng2000/reuse-audio.git
cd reuse-audio
python3 Scripts/prepare_streaming_model.py
bash Scripts/run_streaming.sh --input noisy.wav --output enhanced.wav
```

For the offline model, use
`bash Scripts/run_mac.sh --input noisy.wav --output enhanced.wav`.
The tools protect existing outputs unless `--force` is specified. Their builds
use Xcode so MLX's Metal shaders are compiled. See [development](docs/development.md)
and [the streaming iPhone example](Examples/StreamingApp/README.md).

## Measured performance

Specialized FP32 Metal kernels reduced matched short-test processing on an
**iPhone 13 Pro Max** from **27.65 to 13.01 ms/frame at eight layers**, and from
**39.96 to 17.20 ms at twelve layers**. Reverse-order tests confirmed the gain;
paired STOI changed by less than 4e-9 on the diagnostic recording.

These are measurements of the research prototype. The public package exposes
the kernels as an opt-in path for eight and twelve layers:

```swift
let enhancer = try StreamingEnhancer(
    modelURL: checkpointURL, layers: 12, optimization: .specializedMetal)
```

The twelve-layer one-minute replay averaged **16.98 ms**, with **17.27 ms p95**;
7 of 3,001 frames exceeded 20 ms. This measures processing, not complete call
latency. See [implementation and evidence](docs/metal-optimization/README.md),
the [detailed technical report](docs/metal-optimization/REPORT.md),
and the [earlier four-layer benchmark](docs/streaming-validation.md).

The **30-layer offline model** has a separate experimental port of these ideas.
It preserved diagnostic audio but showed **no reliable speedup on M4 Pro**;
the original offline implementation remains the default. See the
[offline measurements and reproduction steps](docs/offline-metal-optimization.md).

## Licenses and attribution

The products have **different licenses**. [OpenMDW-1.1](LICENSE) covers the
offline component. [NVIDIA NSCLv1](Research/LICENSE.txt) covers the streaming
model and derivative port; it cannot be included in a paid app under that license.
See [NOTICE](NOTICE) for the boundary and required attribution.

This independent Apple port uses MLX/Metal; it is not PyTorch's MPS backend or
an NVIDIA/Apple product. Model weights and personal recordings are excluded
from Git. If used in research, please [cite the port and original papers](docs/citation.md).
