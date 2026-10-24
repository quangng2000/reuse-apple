# RE-USE Apple

**Enhance speech locally on Apple silicon.** A Swift / MLX / Metal port of
[NVIDIA NemotronLabs RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE),
with a Mac command-line tool and an iPhone/iPad recording app.

- One shared inference engine; audio stays on your device.
- 9.61M parameters, FP32, a 38.6 MB checkpoint downloaded separately.
- Import audio, enhance speech, and save a 16 kHz mono WAV.
- Fused Metal Mamba scan; no CUDA, PyTorch, or server at inference time.

This uses **MLX and Metal**, not PyTorch's MPS backend. It is an independent
Apple port, not an NVIDIA or Apple product.

## Run on Mac

Requires an Apple-silicon Mac, macOS 14+, Xcode 16.4+ with its command-line tools,
and Python 3 for model setup.

```sh
git clone https://github.com/quangng2000/reuse-apple.git
cd reuse-apple
bash Scripts/run_mac.sh --input /path/noisy.wav --output /path/enhanced.wav
```

The first run downloads and verifies the pinned model, resolves the pinned Swift
packages, and builds the tool. Add `--benchmark` for three timed inference passes.
Existing output files are protected unless you pass `--force`.

## Run on iPhone or iPad

```sh
python3 Scripts/prepare_model.py
open ReusePocket.xcodeproj
```

Select the **ReusePocket** scheme and a connected device running iOS/iPadOS 17+.
Choose your development team and a unique bundle identifier in Signing & Capabilities,
then build and run. Record up to 30 seconds or import up to 60 seconds; keep the
app in the foreground while enhancing.

## Compatibility and validation

Mac GPU inference is verified on an **M4 Pro**. The iPhone/iPad target is built
for ARM64, but physical iPhone and iPad performance has **not** been validated.
This is shared Apple-platform code, not a guarantee for every device or Intel Mac.

The prototype processed a four-second clip in about **2.3 seconds** on M4 Pro.
That is a saved-recording benchmark, not a live-streaming or phone-speed claim.
See [validation details](docs/validation.md), [architecture](docs/architecture.md),
and [development checks](docs/development.md).

## Model, license, and citation

The port and NVIDIA-derived architecture use [OpenMDW-1.1](LICENSE).
The pinned NemotronLabs model permits commercial use; retain the license and
required attribution notices when redistributing it.
The checkpoint, [model card](Resources/MODEL-CARD.md), and
[third-party notices](Resources/THIRD-PARTY-NOTICES.txt) retain their original attribution.
Weights, personal recordings, and experimental fixtures are not stored in Git.

If this is useful in research, cite this repository and the
[RE-USE paper](https://arxiv.org/abs/2603.02641). See [citation details](docs/citation.md)
and [CITATION.cff](CITATION.cff).
