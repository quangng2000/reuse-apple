# Validation scope

## Existing prototype evidence

The port was checked on an Apple M4 Pro using the pinned NemotronLabs checkpoint,
FP32, MLX Swift 0.29.1, and a four-second 16 kHz mono crackling recording.

- Input STOI: 0.538943.
- Swift/Metal output STOI: 0.880594; PyTorch CPU reference: 0.880562.
- Model-only reconstructed-waveform RMSE with identical input coefficients:
  approximately 5.9e-8.
- Complete waveform inference RMSE against the CPU reference: approximately 8.8e-5.
- Recent GPU inference runs: 2.25–2.30 seconds per four-second clip. Timing excludes
  model loading, audio-file I/O, and metric computation.
- Earlier measured MLX peak active allocation: about 807 MiB, excluding other app
  memory. This is a Mac observation, not an iPhone memory requirement.

Native FFT phase branches can differ near reflection boundaries. Full waveform
inference is not bitwise identical to the PyTorch reference. These results cover
one diagnostic clip and do not establish aggregate enhancement quality.

The diagnostic audio is not redistributed here. Its input SHA-256 was
`7bd6591f0991b4a0937fed38a37bdf193682727232361d3141f0a0c73dc4ddad`.

## Platform status

- M4 Pro Mac: actual GPU inference and numerical checks performed.
- iPhone/iPad: shared engine and an ARM64 iOS app target; unsigned device builds
  checked. Physical installation, speed, memory, and thermal behavior remain untested.
- Other Macs and Apple mobile devices: no benchmark claim.

The repository's generated-audio smoke test checks execution, sample rate,
duration, and finite samples. It is not a speech-quality benchmark. See
[development checks](development.md) for the commands and packaged-build results.
