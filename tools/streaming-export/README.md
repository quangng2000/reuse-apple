# Real-time RE-USE research export

**NSCLv1: noncommercial research and education only.** This is the separate
NVIDIA Real-time RE-USE checkpoint, not the offline30-layer commercial model.
Keep these models and this derived component out of commercial/paid use under
the supplied license. Preserve `LICENSE`, `NOTICE` and `MAMBA-LICENSE.txt` when
redistributing this research component.

The complete pinned checkpoint has13,475,815 FP32 parameters and SHA256
`6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade`.
Source revision:`4c0c186272ca853cd18d832fc5c28777b62f32d1`.
Each export retains the matching learned exit head and first3..12 blocks.
Default depth is4; lookahead is always0. Selected learned weights remain
byte-for-byte FP32: there is no training, quantization or approximation.

## Reproduce from the repository root

```sh
python3.12 -m venv tools/streaming-export/.venv
tools/streaming-export/.venv/bin/python -m pip install -r tools/streaming-export/requirements.txt
tools/streaming-export/.venv/bin/python tools/streaming-export/prepare.py
tools/streaming-export/.venv/bin/python tools/streaming-export/export_streaming.py --layers 3 4 5 6 7 8 9 10 11 12
tools/streaming-export/.venv/bin/python -m unittest discover -s tools/streaming-export -p 'test_*.py' -v
tools/streaming-export/.venv/bin/python tools/streaming-export/parity.py --layers 3 4 5 6 7 8 9 10 11 12
```

`generated/` is ignored and contains downloaded source/checkpoint, ONNX graphs,
manifests and synthetic fixtures. No model weights belong in source control.
`prepare.py` verifies the pinned public source/checkpoint hashes, without loading
the CUDA launcher. `oracle.py` uses the original NVIDIA model and online wrapper
with the separately attributed eager CPU Mamba adapter, not the ONNX builder.

## Runtime contract

Every fixed-depth ONNX graph has two FP32 spectral inputs `magnitude` and `phase`,
each `[1,321,1]`, and matching `enhanced_magnitude`/`enhanced_phase` outputs.
Magnitude is raw compressed log magnitude; phase is radians. Use the existing
output ReLU/expm1, sweep suppression and weighted overlap/add outside the graph.

The manifest lists every persistent FP32 state in canonical order:

- `cache_<key>` input and `next_<key>` output; dots in keys become underscores.
- Initial encoder cache:`[1,2,2,323]`.
- Four encoder dense caches:`[1,64*(i+1),2*(1<<i),323]`.
- Per temporal Mamba:filter`[161,256,3]`, SSM`[161,256,16]`.
- Selected magnitude decoder, then phase decoder:four dense caches with the
  encoder layout but frequency161, plus up-convolution`[1,64,2,322]`.

There are`15+2*layers` persistent caches. Initialize every cache to zero once,
carry every output to the next hop, and clear all caches on reset. Frequency
Mamba's two161-bin scans per block restart for each spectral frame; temporal
state is updated once per hop. Standard ONNX opset17/IR9, CPU ORT1.23.2, no custom
ONNX operators or reduced-precision provider is required.

The native stream frames16kHz mono audio with window640/hop320. It buffers the
first640 samples, suppresses the first synthesis hop, reflects only at the actual
EOF, and trims the flushed output to exactly the total input sample count.
Streaming does not use the offline64k/8k windows or whole-recording peak gain.

## Validation and portable fixtures

All advertised depths3..12 run carried-hop CPU/ORT parity, state shape/finite
checks, every-cache comparison, zero-state reset equality, a proof that carried
state changes output, and actual640-sample reconstructed waveform checks.
Depths4/8 additionally cover8 spectral hops and640/641/959/960/1377/8000/16000
sample audio, including non-hop-aligned exact tails and arbitrary chunk plans.

Each `generated/parity/lN/parity.json` contains:

- `spectral`:frame-major321-float files for input/output magnitude/phase;
- `fixtures`:raw little-endian FP32 input/output audio with sample counts and
  chunk-size plans; outputs come from the independent NVIDIA CPU reference;
- `files`:exact byte sizes, SHA256 and shapes for every fixture;
- raw magnitude/phase drift, all-state error, reset/carry results and fixed
  waveform bound0.0003 (state atol0.0003/rtol0.0002).

These are numerical functional checks on synthetic signals, not a perceptual
quality assessment, Android-device latency measurement or real-time guarantee.
Apple's specialized Metal kernels and phone timings do not establish Android
CPU performance. Physical-device deadline/thermal tests remain separate.
