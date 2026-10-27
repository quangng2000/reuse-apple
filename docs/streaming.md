# Streaming research API

`ReuseStreaming` ports NVIDIA's separate **Real-time RE-USE** model. It is not
the offline 30-layer model with layers removed. The pinned multi-exit checkpoint
has 13,475,815 parameters; the default four-layer, zero-lookahead path uses
2,032,067. The full checkpoint is downloaded separately and verified by hash.
This component uses the [NVIDIA noncommercial research license](../apple/Research/LICENSE.txt).

Apple uses the Swift API below. Android has a separate Kotlin/ONNX Runtime CPU
implementation in [`android/reuse-streaming`](../android/reuse-streaming/README.md),
published as `io.github.quangng2000:reuse-streaming-android:0.1.3`. Both carry the
model's convolution and temporal state across hops. The Android port uses the
same pinned checkpoint and FP32 parameters; small operator/FFT rounding
differences are tested numerically. The iPhone Metal timings below do not apply
to Android. Separate Android physical-phone experiments are documented in the
[Android benchmark results](../android/benchmarks/streaming/results/README.md).
Android 0.1.3 also exposes an explicit eight-layer FP32 Vulkan v2 factory; its
pinned native GPU runtime and model/shader bundle are supplied separately from
the Maven SDK. The ordinary constructor remains CPU-backed.

## Incremental inference

```swift
import ReuseStreaming

let enhancer = try StreamingEnhancer(modelURL: checkpointURL, layers: 4)
for chunk in chunksOfMono16kAudio {
    let restored = try enhancer.push(chunk)
    // Consume the available restored samples in sequence.
}
let tail = try enhancer.finish()
// Consume tail, then reset before starting a new recording.
enhancer.reset()
```

- Input and output are finite `[Float]` samples at **16 kHz, mono**.
- Chunks may have arbitrary lengths. They need not align to 320 samples.
- The model processes 640-sample windows with a 320-sample hop (40/20 ms).
- Startup waits for 640 samples; a push can return no samples yet.
- State persists across pushes. Call `finish()` once at the actual end of audio;
  it flushes the reflected ending and preserves the total sample count.
- Streams shorter than 640 samples are rejected. After finishing, call `reset()`
  before reusing the instance for a different stream.
- Use one serial caller per instance, on a background worker. Never run inference
  directly inside an audio render callback. This API is not a call transport.

The default uses optimized FP32 Metal kernels with the original MLX transforms.
`optimized: false` selects the baseline network implementation for comparison.
Depths 3–12 are supported. Increasing depth changes cost and quality.

For eight or twelve layers, select `optimization: .specializedMetal` to use
prepared frequency values, fused normalization and parallel temporal recurrence.
This retains FP32 and zero lookahead, with small floating-point rounding changes.
It requires `optimized: true`; other depths reject this option. The default
`.standard` preserves the existing configuration, including pooled frequency
processing and compilation at eight layers. See the
[specialized kernel guide](metal-optimization/README.md) for benchmark scope.

## Examples

The [Mac tool](../apple/Examples/StreamingCLI) reads a recording and feeds it to the
streaming API. The [iPhone example](../apple/Examples/StreamingApp) records your voice,
enhances arriving audio, and saves original/restored recordings after stopping.
Use your own development team and unique bundle identifier when signing.
No account, private recording or paid-app purchase configuration is required.

```sh
python3 apple/scripts/prepare_streaming_model.py
bash apple/scripts/run_streaming.sh --input noisy.wav --output restored.wav
```

Use `--benchmark` for processing measurements on the Mac. Specify `--model PATH`
for a checkpoint stored elsewhere, and `--force` to replace an output deliberately.
Add `--baseline` to disable the optimizations for a same-model waveform comparison.
Use the same input/checkpoint, without gain matching or time alignment, when
checking numerical preservation. See each example's help for available flags.

The library supplies enhancement only. Microphone routing, sample-rate conversion,
echo cancellation, buffering, playback, codec and network integration belong to
the host app. The measured phone prototype received microphone callbacks up to
104 ms long despite requesting 20 ms, so processing time alone is not total call
latency. See [measured results](streaming-validation.md).
