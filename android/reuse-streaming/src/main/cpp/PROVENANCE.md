# Native DSP provenance

`pocketfft_hdronly.h` is a byte-for-byte copy of the pinned offline DSP header.
SHA-256: `b15ca29cf47025e76c966d0c5a061e9b56e3e2b312cb05e993ee696f8d9812e5`.

`reuse_audio_dsp.cpp` is copied from `android/reuse-audio/src/main/cpp/reuse_audio_dsp.cpp`.
Original SHA-256: `c427ac5de6996fc5250ad9186818e8a70eaee1ad8bb03ba06fbf9e99683a2a53`.
Streaming SHA-256: `cddbca32295a7b9813496500574296240fb39ac856189c92a05ffaeaaf13ffbc`.
The only source change replaces both JNI function symbol prefixes
`Java_io_riftco_reuseaudio_NativeFourier640_` with
`Java_io_riftco_reusestreaming_StreamingFourier640_`.
FFT math, scaling, validation and precision flags are unchanged.
The separate library target is `reusestreaming_dsp`.

The coefficient file `PeriodicHann640.kt` is the exact offline coefficient array,
with only its Kotlin package renamed. The window is periodic Hann, not sqrt-Hann.

See the bundled MIT adapter license and BSD 3-Clause PocketFFT license.
