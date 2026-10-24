# Shared Apple inference engine

The Mac command-line tool and the iOS/iPadOS app compile the same `Core/` sources.
Swift handles audio files and application state. MLX schedules array operations;
custom Metal kernels implement PReLU and the Mamba selective scan on the GPU.
There is no PyTorch dependency and no call to PyTorch's MPS backend.

The supported checkpoint is NVIDIA NemotronLabs RE-USE, revision
`31f82f2efad1095034492f236e8dac9239ba713b`. It has 9,607,747 parameters.
The runtime checks the checkpoint SHA-256 before loading it. This port uses the
complete released enhancement model; it is not a separate GAN correction head.

## Audio path

1. Read mono or stereo audio and convert it to 16 kHz mono when necessary.
2. Compute a centered STFT with a periodic Hann window: FFT/window 640, hop 80.
3. Encode log1p magnitude and phase, then run 30 bidirectional time/frequency
   Mamba blocks and the magnitude/phase decoders.
4. Apply the official sweep-suppression rule and reconstruct the waveform.
5. Save the converted 16 kHz sample count, preserving the recording's duration.

FP32 is deliberate. The prototype's naive FP16 experiment produced nonfinite
values in later layers; the command-line tool and app use FP32.

## Recordings and device limits

The engine processes four-second windows with half-second overlap and crossfades.
Bidirectional context and normalization restart for each window. Long-recording
results therefore need not equal full-recording inference in the upstream model.
This is offline enhancement, not low-latency streaming.

Convolution tiles retain exact halos to bound temporary memory. The scan retains
16 FP32 state values per channel. Cancellation is checked between model blocks
and windows. The app cancels its work when backgrounded.

Platform APIs and Metal support still matter. The Mac target requires Apple
silicon. The app targets both iPhone and iPad, but memory, processing speed, and
thermal behavior must be measured on each device before claiming support.
