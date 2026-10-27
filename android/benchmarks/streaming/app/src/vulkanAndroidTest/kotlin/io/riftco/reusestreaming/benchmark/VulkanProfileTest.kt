package io.riftco.reusestreaming.benchmark

import android.content.Intent
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.riftco.reusestreaming.ExperimentalVulkanStreaming
import io.riftco.reusestreaming.StreamingEnhancer
import io.riftco.reusestreaming.StreamingEnhancerConfiguration
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/** Short diagnostic only. Timestamp instrumentation is excluded from normal performance calls. */
@RunWith(AndroidJUnit4::class)
class VulkanProfileTest {
    @Test(timeout = 37_000L)
    fun profileEightLayerGpu() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val args = InstrumentationRegistry.getArguments()
        val runId = requireNotNull(args.getString("vulkanRunId"))
        require(runId.matches(Regex("[A-Za-z0-9_.-]{1,80}")))
        val emulator = Build.FINGERPRINT.contains("generic") || Build.MODEL.contains("sdk", true) || Build.HARDWARE.contains("ranchu")
        require(args.getString("vulkanRequirePhysical") == "false" || !emulator)
        require(context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0)
        val start = SystemClock.elapsedRealtime()
        fun checkTime() = check(SystemClock.elapsedRealtime() - start < 30_000L && !Thread.currentThread().isInterrupted)
        val destination = File(checkNotNull(context.getExternalFilesDir(null)), "firebase-vulkan-profile").apply { mkdirs() }
        val result = JSONObject().put("schemaVersion", 1).put("runId", runId).put("success", false)
            .put("layers", 8).put("precision", "FP32").put("sampleRate", 16000).put("hopSamples", 320)
            .put("device", JSONObject().put("model", Build.MODEL).put("sdk", Build.VERSION.SDK_INT)
                .put("fingerprint", Build.FINGERPRINT).put("emulatorReported", emulator))
        val activity = instrumentation.startActivitySync(Intent().setClassName(context.packageName, "${context.packageName}.MainActivity")
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK).putExtra("runId", runId).putExtra("profileHostOnly", true))
        try {
            val bundle = File(context.filesDir, "vulkan-profile-bundle")
            fun copyTree(source: String, target: File) {
                val entries = requireNotNull(context.assets.list(source)); require(entries.isNotEmpty()); target.mkdirs()
                for (name in entries) {
                    checkTime()
                    if (!context.assets.list("$source/$name").isNullOrEmpty()) copyTree("$source/$name", File(target, name))
                    else context.assets.open("$source/$name").use { input -> File(target, name).outputStream().buffered().use { input.copyTo(it, 128 * 1024) } }
                }
            }
            copyTree("vulkan", bundle)
            val inputBytes = context.assets.open("validation/vulkan-benchmark.f32").use { it.readBytes() }
            require(hash(inputBytes) == "e3150c5d89ff157b323627de4e5ede75a35be2e44088b284daa9d47268ccb9ca")
            val input = ByteBuffer.wrap(inputBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().let { buffer -> FloatArray(buffer.remaining()).also { buffer.get(it) } }
            val library = File(context.applicationInfo.nativeLibraryDir, "libreuse_streaming_vulkan.so")
            val configuration = StreamingEnhancerConfiguration(8, 1)
            val profiles = JSONArray(); val profileTimes = JSONArray(); val plainTimes = JSONArray()
            result.put("bundleManifestSha256", hash(File(bundle, "manifest.json").readBytes()))
                .put("nativeLibrarySha256", hash(library.readBytes()))
            var reference: FloatArray? = null
            fun run(enhancer: StreamingEnhancer, profiled: Boolean): FloatArray {
                enhancer.reset(); enhancer.warmUp(); val outputs = ArrayList<Float>()
                outputs.addAll(enhancer.push(input.copyOfRange(0, 640)).toList())
                var offset = 640
                repeat(16) { index ->
                    checkTime(); val hop = input.copyOfRange(offset, offset + 320); offset += 320
                    val t = SystemClock.elapsedRealtimeNanos(); val output = enhancer.push(hop)
                    val elapsed = (SystemClock.elapsedRealtimeNanos() - t) / 1e6
                    require(output.size == 320 && output.all(Float::isFinite)); outputs.addAll(output.toList())
                    if (index >= 4) {
                        if (profiled) { profileTimes.put(elapsed); profiles.put(JSONObject(checkNotNull(enhancer.backendProfileJson))) }
                        else plainTimes.put(elapsed)
                    }
                }
                return outputs.toFloatArray()
            }
            ExperimentalVulkanStreaming.createTiled(bundle, library, configuration).use { engine ->
                result.put("vulkan", JSONObject(checkNotNull(engine.backendDiagnosticsJson)))
                reference = run(engine, false)
            }
            checkTime()
            ExperimentalVulkanStreaming.createProfiled(bundle, library, configuration, tiled = true).use { engine ->
                result.put("profiledVulkan", JSONObject(checkNotNull(engine.backendDiagnosticsJson)))
                val actual = run(engine, true); val expected = checkNotNull(reference)
                require(actual.size == expected.size && actual.indices.all { actual[it].toRawBits() == expected[it].toRawBits() }) { "Profiling changed the GPU output." }
            }
            result.put("profiledOutputBitwiseEqual", true).put("warmupHops", 4).put("timedFrames", 12)
                .put("unprofiledPushMs", plainTimes).put("profiledPushMs", profileTimes).put("profiles", profiles).put("success", true)
        } catch (e: Throwable) {
            result.put("error", JSONObject().put("type", e.javaClass.name).put("message", e.message).put("trace", e.stackTraceToString().take(10000)))
            throw e
        } finally {
            result.put("elapsedMs", SystemClock.elapsedRealtime() - start)
            val bytes = result.toString(2).toByteArray(); File(destination, "$runId-result.json").writeBytes(bytes)
            File(destination, "$runId-result.sha256").writeText(hash(bytes) + "\n")
            instrumentation.runOnMainSync { activity.finish() }
        }
    }
    private fun hash(raw: ByteArray) = MessageDigest.getInstance("SHA-256").digest(raw).joinToString("") { "%02x".format(it.toInt() and 255) }
}
