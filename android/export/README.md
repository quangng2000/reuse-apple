# Commercial 30-layer FP32 Android export

This exports the **offline** NVIDIA RE-USE model already used by `ReuseAudio`.
It does not export the separate, noncommercial `ReuseStreaming` model.

The source is NVIDIA-NemotronLabs-RE-USE revision
`31f82f2efad1095034492f236e8dac9239ba713b`, under OpenMDW-1.1.
The checkpoint must have SHA-256
`06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1`.
All 1,416 original learned tensors / 9,607,747 parameters remain FP32 and
unchanged. There are 30 time-frequency blocks and 120 Mamba selective scans.

## Reproduce

Use Python 3.12 and an isolated environment. Export is CPU-only and never imports
`mamba_ssm`, runs a model download, or needs CUDA.

```sh
python3.12 -m venv android/export/.venv
android/export/.venv/bin/python -m pip install -r android/export/requirements.txt
android/export/.venv/bin/python android/export/export_model.py --checkpoint /absolute/path/model.safetensors
android/export/.venv/bin/python -m unittest discover -s android/export -p test_export.py -v
android/export/.venv/bin/python android/export/parity.py --checkpoint /absolute/path/model.safetensors --model android/export/generated/reuse-30-fp32.onnx
```

Generated weights, raw audio fixtures and validation receipts stay under ignored
`generated/`. The exporter writes a JSON manifest beside the ONNX file with the
model hash, byte size, source identity, precision and audio contract. Verify that
manifest against a trusted application pin before loading an external artifact.

The optional independent source-wiring check uses the separately pinned NVIDIA
modules and the prior eager CPU Mamba reference, without modifying either:

```sh
android/export/.venv/bin/python android/export/upstream_parity.py --checkpoint /absolute/path/model.safetensors --reference-root /absolute/path/Research/Reference --output android/export/generated/upstream-parity.json
```

`validated-artifact.json` records the current artifact identity, observed errors,
quality metrics and limitations. `generate_hann.py` verifies the sealed FP32 Hann
coefficients used by the Android native transform. `pocketfft_proof.py` compiles a
host harness for the exact vendored PocketFFT revision and checks forward/inverse
FFT against PyTorch at 9, 101 and 801 frames. Its flags intentionally retain
expression-local **FP32 FMA** (`-ffp-contract=on`) while disabling fast-math
reassociation. Disabling FMA creates phase-branch drift on the ARM reference.

## Runtime contract

- Inputs: `magnitude`, `phase`, both `float32[1,321,frames]`.
- Outputs: `enhanced_magnitude`, `enhanced_phase`, same shape and dtype.
- Magnitude is **compressed log magnitude**, before output ReLU/expm1 and the
  existing sweep-artifact suppression. Phase is radians.
- ONNX opset 17 / IR 9; standard operators only. Mamba uses sequential `Scan`
  state updates, retaining 16 FP32 states per channel and full bidirectional
  context. No parallel recurrence reassociation, quantization or state truncation.
- Baseline runtime: ONNX Runtime **1.23.2 CPUExecutionProvider**. NNAPI, GPU and
  reduced-precision providers are deliberately outside this validated baseline.

The native wrapper supplies the existing 16 kHz mono transform: FFT 640, hop 80,
periodic Hann, centered reflect padding, unnormalized FFT, log1p(abs) and atan2.
It preserves 64,000-sample windows, 8,000-sample crossfades and the original
variable-length final-window padding (at least 640 samples, rounded to hop 80).
Always padding a short final window to four seconds changes bidirectional context
and instance-normalization statistics, so it is not an equivalent optimization.
The library does not add the application's separate global peak gain.

## What parity establishes

`parity.py` checks the exact graph hash, every original weight byte, all 120 scans,
finite outputs and dynamic frame lengths. It compares all-layer spectral output
and reconstructed audio against an eager PyTorch CPU oracle on deterministic
synthetic inputs. It records max absolute error, RMSE, shape, runtime and whether
the results happen to be bitwise equal. The fixed waveform max-error bound is
0.0003; paired speech additionally requires STOI delta <=0.0001 and SI-SDR delta
<=0.02 dB. A failed waveform/quality check stops the script. Raw spectral/wrapped
phase max error 0.003 remains a separately flagged diagnostic, not an audio
acceptance gate: exact silence can amplify constant-channel normalization
roundoff, and raw atan2 phase is ill-conditioned at nearly-real/near-zero bins.
All such raw drift remains visible in the receipt, including a false phase flag.
The native Android tests separately compare the exported raw FP32 fixtures.
`compare_pcm.py` applies the same unchanged waveform and paired-quality bounds to
an explicitly supplied native FP32 output, keeping end-to-end DSP validation
separate from inference with already-computed spectra.

This is numerical FP32 parity, **not bitwise cross-runtime identity**, a speech
quality evaluation, an Android hardware latency measurement or a real-time claim.
Convolution/reduction/transcendental implementations can round differently.
The standard CPU scan is a correct offline baseline, not a promised mobile
accelerator implementation. Benchmark physical Android hardware before making
performance or memory claims.

Primary operator/runtime references:
[ONNX Scan](https://onnx.ai/onnx/operators/onnx__Scan.html),
[ONNX Runtime Android](https://onnxruntime.ai/docs/tutorials/mobile/deploy-android.html).
