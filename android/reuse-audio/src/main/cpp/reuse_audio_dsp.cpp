// Real FP32 FFT adapter for the pinned PyTorch PocketFFT implementation.
// Original adaptation: SPDX-License-Identifier: MIT.
// PocketFFT's separate BSD-3-Clause notice is retained beside its header.
#include <jni.h>
#include <array>
#include <cmath>
#include <complex>
#include <exception>
#include "pocketfft_hdronly.h"

namespace {
constexpr size_t kSize = 640;
constexpr size_t kBins = kSize / 2 + 1;

void fail(JNIEnv* env, const char* kind, const char* message) {
    jclass type = env->FindClass(kind);
    if (type != nullptr) env->ThrowNew(type, message);
}

bool read(JNIEnv* env, jfloatArray real, jfloatArray imag,
          std::array<float, kSize>& r, std::array<float, kSize>& i) {
    if (real == nullptr || imag == nullptr || env->GetArrayLength(real) != kSize ||
        env->GetArrayLength(imag) != kSize || env->IsSameObject(real, imag)) {
        fail(env, "java/lang/IllegalArgumentException", "FFT arrays must be distinct and each contain exactly 640 FP32 samples");
        return false;
    }
    env->GetFloatArrayRegion(real, 0, kSize, r.data());
    env->GetFloatArrayRegion(imag, 0, kSize, i.data());
    return !env->ExceptionCheck();
}

void write(JNIEnv* env, jfloatArray real, jfloatArray imag,
           const std::array<float, kSize>& r, const std::array<float, kSize>& i) {
    env->SetFloatArrayRegion(real, 0, kSize, r.data());
    if (!env->ExceptionCheck()) env->SetFloatArrayRegion(imag, 0, kSize, i.data());
}
}

extern "C" JNIEXPORT void JNICALL
Java_io_riftco_reuseaudio_NativeFourier640_forward(JNIEnv* env, jclass, jfloatArray real, jfloatArray imag) {
    try {
        std::array<float, kSize> r{}, i{};
        if (!read(env, real, imag, r, i)) return;
        for (size_t index = 0; index < kSize; ++index) {
            if (!std::isfinite(r[index]) || i[index] != 0.0f) {
                fail(env, "java/lang/IllegalArgumentException", "Forward FFT requires finite real input and zero imaginary input");
                return;
            }
        }
        std::array<std::complex<float>, kBins> spectrum{};
        pocketfft::r2c<float>({kSize}, {sizeof(float)}, {sizeof(std::complex<float>)}, 0,
                            true, r.data(), spectrum.data(), 1.0f, 1);
        for (size_t bin = 0; bin < kBins; ++bin) {
            r[bin] = spectrum[bin].real();
            i[bin] = spectrum[bin].imag();
        }
        for (size_t bin = kBins; bin < kSize; ++bin) {
            r[bin] = spectrum[kSize - bin].real();
            i[bin] = -spectrum[kSize - bin].imag();
        }
        write(env, real, imag, r, i);
    } catch (const std::exception& error) {
        fail(env, "java/lang/IllegalStateException", error.what());
    } catch (...) {
        fail(env, "java/lang/IllegalStateException", "Native FP32 forward FFT failed");
    }
}

extern "C" JNIEXPORT void JNICALL
Java_io_riftco_reuseaudio_NativeFourier640_inverse(JNIEnv* env, jclass, jfloatArray real, jfloatArray imag) {
    try {
        std::array<float, kSize> r{}, i{};
        if (!read(env, real, imag, r, i)) return;
        std::array<std::complex<float>, kBins> spectrum{};
        for (size_t bin = 0; bin < kBins; ++bin) {
            if (!std::isfinite(r[bin]) || !std::isfinite(i[bin])) {
                fail(env, "java/lang/IllegalArgumentException", "Inverse FFT requires finite spectral input");
                return;
            }
            spectrum[bin] = {r[bin], i[bin]};
        }
        // The real inverse intentionally ignores the redundant mirrored half,
        // just as torch.istft consumes the one-sided 321-bin complex spectrum.
        pocketfft::c2r<float>({kSize}, {sizeof(std::complex<float>)}, {sizeof(float)}, 0,
                            false, spectrum.data(), r.data(), 1.0f / static_cast<float>(kSize), 1);
        i.fill(0.0f);
        write(env, real, imag, r, i);
    } catch (const std::exception& error) {
        fail(env, "java/lang/IllegalStateException", error.what());
    } catch (...) {
        fail(env, "java/lang/IllegalStateException", "Native FP32 inverse FFT failed");
    }
}
