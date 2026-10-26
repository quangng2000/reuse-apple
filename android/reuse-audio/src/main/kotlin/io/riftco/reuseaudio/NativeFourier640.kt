package io.riftco.reuseaudio

/** Pinned PocketFFT<float>, shared by Android JNI and the host parity harness. */
internal object NativeFourier640 {
    init { System.loadLibrary("reuseaudio_dsp") }
    @JvmStatic external fun forward(real: FloatArray, imaginary: FloatArray)
    @JvmStatic external fun inverse(real: FloatArray, imaginary: FloatArray)
}
