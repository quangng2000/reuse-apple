# RE-USE Apple

**Speech enhancement on Apple silicon, as a standalone Swift Package.**
Use native Swift APIs backed by MLX and Metal. Audio stays on the device;
inference needs no Python, CUDA, or server.

- **`ReuseAudio`** — offline enhancement with NVIDIA NemotronLabs RE-USE.
  9.61M parameters, FP32, OpenMDW-1.1 license; commercial use permitted.
- **`ReuseStreaming`** — incremental enhancement with NVIDIA Real-time RE-USE.
  Four-layer default, FP32, 16 kHz mono; **noncommercial research only**.
- Mac command-line tools and iPhone/iPad examples consume the library APIs.

## Add the package

In Xcode, add `https://github.com/quangng2000/reuse-apple` as a package dependency
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
git clone https://github.com/quangng2000/reuse-apple.git
cd reuse-apple
python3 Scripts/prepare_streaming_model.py
bash Scripts/run_streaming.sh --input noisy.wav --output enhanced.wav
```

For the offline model, use
`bash Scripts/run_mac.sh --input noisy.wav --output enhanced.wav`.
The tools protect existing outputs unless `--force` is specified. Their builds
use Xcode so MLX's Metal shaders are compiled. See [development](docs/development.md)
and [the streaming iPhone example](Examples/StreamingApp/README.md).

## Measured performance

The optimized four-layer prototype processed 60 seconds of repeated audio on an
**iPhone 13 Pro Max** at **18.28 ms/frame average**, **19.03 ms p95**. About 1%
of frames exceeded the 20 ms processing budget. Waveform checks preserved the
original output to FP32 rounding differences. This measures processing, not
complete call latency. See [benchmark scope and evidence](docs/streaming-validation.md).

## Licenses and attribution

The products have **different licenses**. [OpenMDW-1.1](LICENSE) covers the
offline component. [NVIDIA NSCLv1](Research/LICENSE.txt) covers the streaming
model and derivative port; it cannot be included in a paid app under that license.
See [NOTICE](NOTICE) for the boundary and required attribution.

This independent Apple port uses MLX/Metal; it is not PyTorch's MPS backend or
an NVIDIA/Apple product. Model weights and personal recordings are excluded
from Git. If used in research, please [cite the port and original papers](docs/citation.md).
