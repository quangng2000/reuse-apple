// Copyright (c) 2026 quangng2000. MIT adapter terms in ADAPTER_LICENSE.txt.
// Transport-only helpers. Existing PocketFFT and neural arithmetic are unchanged.
#include <jni.h>
#include <array>
#include <cstdint>
#include <cstring>

namespace {
constexpr std::size_t kMaximumBuffers = 128;
struct BufferView { const std::uint8_t* data; std::size_t count; };

bool fail(JNIEnv* env, const char* message) {
    jclass type = env->FindClass("java/lang/IllegalArgumentException");
    if (type != nullptr) { env->ThrowNew(type, message); env->DeleteLocalRef(type); }
    return false;
}

bool views(JNIEnv* env, jobjectArray buffers, jintArray counts,
           std::array<BufferView, kMaximumBuffers>& output, jsize& length, bool writable = false) {
    if (buffers == nullptr || counts == nullptr) return fail(env, "Direct FP32 buffers/counts are required.");
    length = env->GetArrayLength(buffers);
    if (length <= 0 || length > static_cast<jsize>(kMaximumBuffers) || env->GetArrayLength(counts) != length)
        return fail(env, "Invalid direct FP32 buffer/count list.");
    std::array<jint, kMaximumBuffers> sizes{};
    env->GetIntArrayRegion(counts, 0, length, sizes.data());
    if (env->ExceptionCheck()) return false;
    jmethodID read_only_method = nullptr;
    if (writable) {
        jclass type = env->FindClass("java/nio/Buffer");
        if (type == nullptr) return false;
        read_only_method = env->GetMethodID(type, "isReadOnly", "()Z");
        env->DeleteLocalRef(type);
        if (read_only_method == nullptr) return false;
    }
    for (jsize i = 0; i < length; ++i) {
        jobject buffer = env->GetObjectArrayElement(buffers, i);
        if (buffer == nullptr) return fail(env, "A direct FP32 buffer is null.");
        const bool read_only = writable && env->CallBooleanMethod(buffer, read_only_method);
        void* address = env->GetDirectBufferAddress(buffer);
        const jlong capacity = env->GetDirectBufferCapacity(buffer);
        env->DeleteLocalRef(buffer);
        if (env->ExceptionCheck()) return false;
        if (read_only || address == nullptr || sizes[i] <= 0 || capacity < static_cast<jlong>(sizes[i]) * 4)
            return fail(env, "Expected a direct ByteBuffer large enough for its FP32 element count.");
        output[i] = { static_cast<const std::uint8_t*>(address), static_cast<std::size_t>(sizes[i]) };
    }
    return true;
}
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_riftco_reusestreaming_NativeStreamingBuffers_allFinite(JNIEnv* env, jclass,
                                                              jobjectArray buffers, jintArray counts) {
    std::array<BufferView, kMaximumBuffers> storage{};
    jsize length = 0;
    if (!views(env, buffers, counts, storage, length)) return JNI_FALSE;
    for (jsize buffer = 0; buffer < length; ++buffer) {
        const BufferView view = storage[buffer];
        std::uint32_t nonfinite = 0;
        for (std::size_t i = 0; i < view.count; ++i) {
            std::uint32_t bits;
            std::memcpy(&bits, view.data + i * 4, sizeof(bits));
            nonfinite |= static_cast<std::uint32_t>((bits & 0x7f800000u) == 0x7f800000u);
        }
        if (nonfinite != 0) return JNI_FALSE;
    }
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_io_riftco_reusestreaming_NativeStreamingBuffers_zero(JNIEnv* env, jclass,
                                                         jobjectArray buffers, jintArray counts) {
    std::array<BufferView, kMaximumBuffers> storage{};
    jsize length = 0;
    if (!views(env, buffers, counts, storage, length, true)) return;
    for (jsize buffer = 0; buffer < length; ++buffer)
        std::memset(const_cast<std::uint8_t*>(storage[buffer].data), 0, storage[buffer].count * 4);
}
