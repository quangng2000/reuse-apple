# Separate research streaming models

These manifests and source pins describe NVIDIA Real-time RE-USE, not the
commercial offline30-layer model. The original54,055,764-byte checkpoint and
derived streaming model use **NSCLv1, noncommercial research/education only**.
See `NSCL-LICENSE.txt` and `source.json`.

`catalog.json` pins each fixed-depth3..12 FP32 ONNX export, its manifest and
synthetic reference receipt. `validated-artifacts.json` adds the exact synthetic
parity archive and license/source-file hashes. The public artifact release is
[`android-streaming-models-v1`](https://github.com/quangng2000/reuse-audio/releases/tag/android-streaming-models-v1).
The default depth is4 and lookahead is0.

The ten trusted JSON manifests are checked into [`manifests/`](manifests/), named
`reuse-streaming-l3-fp32.manifest.json` through `reuse-streaming-l12-fp32.manifest.json`.
Each is byte/hash-verified against the catalog. Bundle the selected trusted
manifest with the consuming research application; download only its matching
ONNX model from the public release and verify its expected size/SHA256 before
loading. A downloaded sidecar is not a replacement for a trusted manifest.

Model weights and generated artifacts stay outside Git and outside the AAR.
The platform-neutral export/reference tooling is in `tools/streaming-export`.
