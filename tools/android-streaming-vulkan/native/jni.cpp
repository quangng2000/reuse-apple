// SPDX-License-Identifier: MIT
// Original native runtime adapter; see LICENSE.txt in this directory.
#include "runtime.h"
#include <jni.h>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace {
std::mutex registryMutex;
std::map<jlong, std::shared_ptr<reuse::vulkan::Engine>> registry;
std::atomic<jlong> nextHandle{1};
void fail(JNIEnv* env, const std::exception& error) {
    if (!env->ExceptionCheck()) env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
}
std::shared_ptr<reuse::vulkan::Engine> engine(jlong handle) {
    std::lock_guard<std::mutex> guard(registryMutex);
    auto entry = registry.find(handle);
    if (entry == registry.end()) throw std::runtime_error("Closed or invalid Vulkan engine handle");
    return entry->second;
}
std::string path(JNIEnv* env, jstring string) {
    if (!string) throw std::runtime_error("Null Vulkan asset path");
    const char* chars = env->GetStringUTFChars(string, nullptr);
    if (!chars) throw std::runtime_error("Cannot read Vulkan asset path");
    std::string result(chars); env->ReleaseStringUTFChars(string, chars);
    return result;
}
}
extern "C" JNIEXPORT jlong JNICALL Java_io_riftco_reusestreaming_NativeVulkanEngine_create(
    JNIEnv* env, jobject, jstring plan, jstring weights, jstring shaders) {
    try {
        auto value = std::make_shared<reuse::vulkan::Engine>(path(env, plan), path(env, weights), path(env, shaders), false);
        auto handle = nextHandle.fetch_add(1);
        if (handle <= 0) throw std::runtime_error("Vulkan handle space exhausted");
        std::lock_guard<std::mutex> guard(registryMutex); registry.emplace(handle, std::move(value)); return handle;
    } catch (const std::exception& error) { fail(env, error); return 0; }
}
extern "C" JNIEXPORT jlong JNICALL Java_io_riftco_reusestreaming_NativeVulkanEngine_createTuned(
    JNIEnv* env, jobject, jstring plan, jstring weights, jstring shaders, jboolean tiledMatmul, jboolean tiledConv) {
    try {
        auto value = std::make_shared<reuse::vulkan::Engine>(path(env, plan), path(env, weights), path(env, shaders), false,
                                                          tiledMatmul == JNI_TRUE, tiledConv == JNI_TRUE);
        auto handle = nextHandle.fetch_add(1);
        if (handle <= 0) throw std::runtime_error("Vulkan handle space exhausted");
        std::lock_guard<std::mutex> guard(registryMutex); registry.emplace(handle, std::move(value)); return handle;
    } catch (const std::exception& error) { fail(env, error); return 0; }
}
extern "C" JNIEXPORT jlong JNICALL Java_io_riftco_reusestreaming_NativeVulkanEngine_createProfiled(
    JNIEnv* env, jobject, jstring plan, jstring weights, jstring shaders, jboolean tiledMatmul, jboolean tiledConv) {
    try {
        auto value = std::make_shared<reuse::vulkan::Engine>(path(env, plan), path(env, weights), path(env, shaders), false,
                                                          tiledMatmul == JNI_TRUE, tiledConv == JNI_TRUE, true);
        auto handle = nextHandle.fetch_add(1);
        if (handle <= 0) throw std::runtime_error("Vulkan handle space exhausted");
        std::lock_guard<std::mutex> guard(registryMutex); registry.emplace(handle, std::move(value)); return handle;
    } catch (const std::exception& error) { fail(env, error); return 0; }
}
extern "C" JNIEXPORT void JNICALL Java_io_riftco_reusestreaming_NativeVulkanEngine_frame(
    JNIEnv* env, jobject, jlong handle, jfloatArray magnitude, jfloatArray phase, jfloatArray outMagnitude, jfloatArray outPhase) {
    try {
        for (auto value : {magnitude, phase, outMagnitude, outPhase}) if (!value || env->GetArrayLength(value) != 321) throw std::runtime_error("Vulkan frame arrays must have 321 elements");
        std::array<float, 321> m{}, p{}, om{}, op{};
        env->GetFloatArrayRegion(magnitude, 0, 321, m.data()); env->GetFloatArrayRegion(phase, 0, 321, p.data());
        if (env->ExceptionCheck()) return;
        engine(handle)->frame(m.data(), p.data(), om.data(), op.data(), 321);
        env->SetFloatArrayRegion(outMagnitude, 0, 321, om.data()); env->SetFloatArrayRegion(outPhase, 0, 321, op.data());
    } catch (const std::exception& error) { fail(env, error); }
}
extern "C" JNIEXPORT void JNICALL Java_io_riftco_reusestreaming_NativeVulkanEngine_commit(JNIEnv* env, jobject, jlong handle) {
    try { engine(handle)->commit(); } catch (const std::exception& error) { fail(env, error); }
}
extern "C" JNIEXPORT void JNICALL Java_io_riftco_reusestreaming_NativeVulkanEngine_reset(JNIEnv* env, jobject, jlong handle) {
    try { engine(handle)->reset(); } catch (const std::exception& error) { fail(env, error); }
}
extern "C" JNIEXPORT jfloatArray JNICALL Java_io_riftco_reusestreaming_NativeVulkanEngine_copyState(JNIEnv* env, jobject, jlong handle, jint index) {
    try {
        auto state = engine(handle)->copyState(size_t(index));
        auto result = env->NewFloatArray(jsize(state.size()));
        if (result) env->SetFloatArrayRegion(result, 0, jsize(state.size()), state.data());
        return result;
    } catch (const std::exception& error) { fail(env, error); return nullptr; }
}
extern "C" JNIEXPORT jstring JNICALL Java_io_riftco_reusestreaming_NativeVulkanEngine_infoJson(JNIEnv* env, jobject, jlong handle) {
    try { return env->NewStringUTF(engine(handle)->infoJson().c_str()); } catch (const std::exception& error) { fail(env, error); return nullptr; }
}
extern "C" JNIEXPORT jstring JNICALL Java_io_riftco_reusestreaming_NativeVulkanEngine_profileJson(JNIEnv* env, jobject, jlong handle) {
    try { const auto profile = engine(handle)->profileJson(); return profile.empty() ? nullptr : env->NewStringUTF(profile.c_str()); }
    catch (const std::exception& error) { fail(env, error); return nullptr; }
}
extern "C" JNIEXPORT jdouble JNICALL Java_io_riftco_reusestreaming_NativeVulkanEngine_lastGpuMilliseconds(JNIEnv* env, jobject, jlong handle) {
    try { return engine(handle)->lastGpuMilliseconds(); } catch (const std::exception& error) { fail(env, error); return -1; }
}
extern "C" JNIEXPORT void JNICALL Java_io_riftco_reusestreaming_NativeVulkanEngine_close(JNIEnv* env, jobject, jlong handle) {
    try {
        std::shared_ptr<reuse::vulkan::Engine> removed;
        { std::lock_guard<std::mutex> guard(registryMutex); auto entry = registry.find(handle); if (entry == registry.end()) return; removed = std::move(entry->second); registry.erase(entry); }
        removed.reset();
    } catch (const std::exception& error) { fail(env, error); }
}
