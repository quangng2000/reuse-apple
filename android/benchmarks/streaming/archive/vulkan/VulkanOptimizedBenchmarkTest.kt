package io.riftco.reusestreaming.benchmark

import android.app.Activity
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.riftco.reusestreaming.*
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.sqrt

/** One bounded physical execution: numerical gates first, then same-phone CPU/GPU/CPU comparison. */
@RunWith(AndroidJUnit4::class)
class VulkanOptimizedBenchmarkTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private lateinit var root: File
    private lateinit var artifacts: File
    private lateinit var runId: String
    private var deadline = 0L

    @Test(timeout = 94_000L)
    fun validateAndCompareEightLayerFP32() {
        val args = InstrumentationRegistry.getArguments()
        runId = args.getString("vulkanRunId") ?: "vulkan-${System.currentTimeMillis()}"
        require(runId.matches(Regex("[A-Za-z0-9_.-]{1,80}")))
        val physical = args.getString("vulkanRequirePhysical") != "false"
        val frames = args.getString("vulkanFrames")?.toInt() ?: 300
        require(frames in 32..300)
        val started = SystemClock.elapsedRealtime()
        deadline = started + 85_000L
        artifacts = File(checkNotNull(context.getExternalFilesDir(null)), "firebase-vulkan-v2").apply { mkdirs() }
        root = File(context.filesDir, "vulkan-validation-v2").apply { mkdirs() }
        val cases = JSONArray()
        val result = JSONObject().put("schemaVersion", 2).put("runId", runId).put("success", false)
            .put("scope", "Foreground synchronous 320-sample public streaming calls, including DSP, GPU completion and output readback; excludes audio-device and network latency")
            .put("layers", 8).put("precision", "FP32").put("hopSamples", 320).put("sampleRate", 16000)
            .put("timedFramesPerCase", frames).put("warmupHops", 32).put("cases", cases)
            .put("device", JSONObject().put("model", Build.MODEL).put("manufacturer", Build.MANUFACTURER)
                .put("fingerprint", Build.FINGERPRINT).put("sdk", Build.VERSION.SDK_INT)
                .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList())).put("emulatorReported", emulator()))
        var activity: Activity? = null
        try {
            require(!physical || !emulator()) { "A physical phone is required." }
            require(context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) { "Release target required." }
            activity = instrumentation.startActivitySync(Intent().setClassName(context.packageName, "${context.packageName}.MainActivity")
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                .putExtra("runId", runId).putExtra("profileHostOnly", true))
            copyTree("vulkan", File(root, "vulkan"))
            copyTree("vulkan-baseline", File(root, "vulkan-baseline"))
            val gpu = ExperimentalVulkanStreaming.createTiled(File(root, "vulkan-baseline"),
                File(context.applicationInfo.nativeLibraryDir, "libreuse_streaming_vulkan.so"), StreamingEnhancerConfiguration(8, 1))
            gpu.use {
                val device = JSONObject(checkNotNull(gpu.backendDiagnosticsJson))
                require(device.getBoolean("hardwareGpu") && !device.getBoolean("softwareDeviceAllowed") &&
                    device.getString("precision") == "float32" && device.getInt("stateCount") == 31 &&
                    device.getBoolean("statePingPong") && device.getBoolean("gpuFiniteChecks") &&
                    !device.getBoolean("perDispatchProfiling"))
                result.put("baselineVulkan", device)
                result.put("baselineBundleManifestSha256", hash(File(root, "vulkan-baseline/manifest.json").readBytes()))
                result.put("bundleManifestSha256", hash(File(root, "vulkan/manifest.json").readBytes()))
                // The native engine selects a real integrated/discrete Vulkan GPU and rejects CPU devices.
                // The immutable baseline has passed the prior real-phone quality gate.
                checkWork()
                ExperimentalVulkanStreaming.createTiled(File(root, "vulkan"),
                    File(context.applicationInfo.nativeLibraryDir, "libreuse_streaming_vulkan.so"), StreamingEnhancerConfiguration(8, 1)).use { tiled ->
                  val optimizedDevice = JSONObject(checkNotNull(tiled.backendDiagnosticsJson))
                  require(!optimizedDevice.getBoolean("perDispatchProfiling") &&
                      optimizedDevice.getBoolean("hardwareGpu") && !optimizedDevice.getBoolean("softwareDeviceAllowed") &&
                      optimizedDevice.getString("precision") == "float32" && optimizedDevice.getInt("stateCount") == 31 &&
                      optimizedDevice.getBoolean("statePingPong") && optimizedDevice.getBoolean("gpuFiniteChecks"))
                  for (key in listOf("deviceName", "vendorId", "deviceId", "driverVersion", "apiVersion")) {
                      require(optimizedDevice.get(key) == device.get(key)) { "GPU changed within the comparison." }
                  }
                  result.put("vulkan", optimizedDevice)
                  result.put("quality", validateQuality(tiled, "optimized"))
                  stable().use { cpu ->
                    val fullInput = asset("validation/vulkan-benchmark.f32")
                    require(hash(fullInput) == "e3150c5d89ff157b323627de4e5ede75a35be2e44088b284daa9d47268ccb9ca")
                    val input = floats(fullInput).copyOfRange(0, 640 + (32 + frames) * 320)
                    result.put("benchmarkInputSha256", hash(bytes(input)))
                    export("benchmark-input", input)
                    cases.put(measure(cpu, "stable-v2-cpu-before", "CPUExecutionProvider", 8, input, frames))
                    cases.put(measure(gpu, "vulkan-tiled-baseline", "Vulkan", 0, input, frames))
                    cases.put(measure(tiled, "vulkan-tiled-optimized", "Vulkan", 0, input, frames))
                    cases.put(measure(cpu, "stable-v2-cpu-after", "CPUExecutionProvider", 8, input, frames))
                  }
                }
            }
            result.put("success", true)
        } catch (failure: Throwable) {
            result.put("error", JSONObject().put("type", failure.javaClass.name)
                .put("message", failure.message ?: "No message").put("trace", failure.stackTraceToString().take(12000)))
            throw failure
        } finally {
            result.put("elapsedMs", SystemClock.elapsedRealtime() - started)
            val output = result.toString(2).toByteArray()
            File(artifacts, "$runId-result.json").writeBytes(output)
            File(artifacts, "$runId-result.sha256").writeText(hash(output) + "\n")
            Log.i("REUSE_VULKAN", "RESULT runId=$runId success=${result.optBoolean("success")} sha256=${hash(output)}")
            activity?.let { a -> instrumentation.runOnMainSync { a.finish() } }
        }
    }

    private fun validateQuality(gpu: StreamingEnhancer, label: String): JSONObject {
        val input = floats(asset("validation/noisy-speech.f32"))
        require(input.size == 64000 && hash(bytes(input)) == "c449b1ebf8f51c80f7530961b41c62a733a1033a60140fc16c0b37186120835f")
        val golden = asset("validation/vulkan-golden-cpu.f32")
        require(hash(golden) == "54ea90bd246e3a527aa3c5ee00f9e8e9a2b10983b53153a6b1fe7427ca16e65f")
        val reference = floats(golden)
        val candidate = render(gpu, input, intArrayOf(640, 320))
        export("quality-$label-input", input); export("quality-$label-cpu", reference); export("quality-$label-vulkan", candidate)
        val errors = compare(reference, candidate)
        require(errors.getDouble("maxAbs") <= 5e-6 && errors.getDouble("relativeRms") <= 1e-5) {
            "Vulkan output failed the predeclared FP32 waveform gate: $errors"
        }
        // Exercise startup reflection, partial chunks, EOF, reset and GPU bank reuse on identical samples.
        val short = input.copyOfRange(0, 1377)
        val first = render(gpu, short, intArrayOf(640, 320))
        val replay = render(gpu, short, intArrayOf(1, 319, 7, 640, 3, 407))
        require(first.size == short.size && first.indices.all { first[it].toRawBits() == replay[it].toRawBits() }) {
            "Vulkan reset/chunk replay differs."
        }
        for (length in listOf(640, 641, 959, 960, 8000, 16000)) {
            checkWork()
            val audio = input.copyOfRange(0, length)
            val actual = render(gpu, audio, intArrayOf(37, 320, 411))
            require(actual.size == length && actual.all(Float::isFinite)) { "Vulkan partial-tail/sample-count failure." }
        }
        val cancelled = StreamingCancellationToken().apply { cancel() }
        gpu.reset()
        try { gpu.push(short, cancelled); error("Expected cancellation") }
        catch (_: java.util.concurrent.CancellationException) { /* no frame submitted */ }
        gpu.reset()
        val afterCancel = render(gpu, short, intArrayOf(640, 320))
        require(first.indices.all { first[it].toRawBits() == afterCancel[it].toRawBits() })
        return errors.put("passed", true).put("samples", input.size).put("resetBitwiseReplay", true)
            .put("partialTailsAndCancellation", true).put("reference", "Pinned published FP32 ONNX output from the prior physical S25 Ultra test, identical synthetic input")
            .put("audioQualityScope", "One synthetic four-second noisy speech diagnostic; not broad perceptual validation")
    }

    private fun measure(enhancer: StreamingEnhancer, label: String, backend: String, threads: Int,
                        input: FloatArray, frames: Int): JSONObject {
        checkWork(); enhancer.reset(); enhancer.warmUp()
        var count = enhancer.push(input.copyOfRange(0, 640)).size
        require(count == 320)
        var offset = 640
        repeat(32) {
            checkWork(); val out = enhancer.push(input.copyOfRange(offset, offset + 320)); offset += 320
            require(out.size == 320 && out.all(Float::isFinite)); count += out.size
        }
        val hops = Array(frames) { i -> input.copyOfRange(offset + i * 320, offset + (i + 1) * 320) }
        val times = DoubleArray(frames); val renderTimes = DoubleArray(frames); var checksum = 0.0
        repeat(frames) { i ->
            checkWork()
            val start = SystemClock.elapsedRealtimeNanos()
            val output = enhancer.push(hops[i])
            times[i] = (SystemClock.elapsedRealtimeNanos() - start) / 1e6
            renderTimes[i] = enhancer.lastFrameMilliseconds.single()
            require(output.size == 320 && output.all(Float::isFinite))
            count += output.size; output.forEach { checksum += it }
        }
        val tail = enhancer.finish(); require(tail.all(Float::isFinite)); count += tail.size
        require(count == input.size)
        val sorted = times.sorted()
        fun percentile(p: Double) = sorted[(ceil(p * sorted.size).toInt() - 1).coerceIn(sorted.indices)]
        val latency = JSONObject().put("mean", times.average()).put("p50", percentile(.5)).put("p95", percentile(.95))
            .put("p99", percentile(.99)).put("min", sorted.first()).put("max", sorted.last())
        Log.i("REUSE_VULKAN", "$label mean=${times.average()} p95=${percentile(.95)}")
        return JSONObject().put("caseId", label).put("backend", backend).put("cpuThreads", threads)
            .put("layers", 8).put("precision", "FP32").put("frames", frames)
            .put("rawPushMs", JSONArray(times.toList())).put("rawRenderMs", JSONArray(renderTimes.toList()))
            .put("latencyMs", latency).put("over20Ms", times.count { it > 20 }).put("rtf", times.average() / 20)
            .put("outputSamples", count).put("inputSamples", input.size).put("checksum", checksum).put("allOutputFinite", true)
    }

    private fun render(enhancer: StreamingEnhancer, input: FloatArray, chunks: IntArray): FloatArray {
        enhancer.reset(); val output = ArrayList<Float>(); var offset = 0; var index = 0
        while (offset < input.size) {
            checkWork(); val end = minOf(input.size, offset + chunks[index++ % chunks.size])
            output.addAll(enhancer.push(input.copyOfRange(offset, end)).toList()); offset = end
        }
        output.addAll(enhancer.finish().toList())
        require(output.size == input.size && output.all(Float::isFinite))
        return output.toFloatArray()
    }
    private fun stable() = ExperimentalNativeStreamingStableV2.create(
        copy("native-v2-stable/reuse-streaming-l8-native-stable-v2.onnx"),
        copy("native-v2-stable/reuse-streaming-l8-native-stable-v2.manifest.json"),
        File(context.applicationInfo.nativeLibraryDir, "libreuse_streaming_stable_v2.so"),
        copy("native-v2-stable/libreuse_streaming_stable_v2.so.json"), StreamingEnhancerConfiguration(8, 8))
    private fun compare(a: FloatArray, b: FloatArray): JSONObject {
        require(a.size == b.size); var maximum = 0.0; var error = 0.0; var signal = 0.0
        a.indices.forEach { i -> val delta = b[i].toDouble() - a[i]; maximum = maxOf(maximum, abs(delta)); error += delta * delta; signal += a[i].toDouble() * a[i] }
        require(signal > 1e-12 && error.isFinite()) { "Invalid or silent quality reference." }
        return JSONObject().put("maxAbs", maximum).put("relativeRms", sqrt(error / maxOf(signal, 1e-30)))
    }
    private fun export(label: String, samples: FloatArray) {
        val pcm = bytes(samples); val file = File(artifacts, "$runId-$label.f32")
        file.writeBytes(pcm); File(artifacts, file.name + ".sha256").writeText(hash(pcm) + "\n")
    }
    private fun asset(name: String) = context.assets.open(name).use { it.readBytes() }
    private fun copyAssetTo(name: String, destination: File): File {
        destination.parentFile!!.mkdirs()
        context.assets.open(name).use { input -> destination.outputStream().buffered().use { output ->
            input.copyTo(output, 128 * 1024)
        } }
        return destination
    }
    private fun copy(name: String): File = copyAssetTo(name, File(root, name))
    private fun copyTree(source: String, destination: File) {
        val entries = context.assets.list(source) ?: error("Missing assets")
        require(entries.isNotEmpty()); destination.mkdirs()
        for (name in entries) {
            if (!context.assets.list("$source/$name").isNullOrEmpty()) copyTree("$source/$name", File(destination, name))
            else copyAssetTo("$source/$name", File(destination, name))
        }
    }
    private fun checkWork() = check(SystemClock.elapsedRealtime() < deadline && !Thread.currentThread().isInterrupted) { "Vulkan validation exceeded 85-second work limit." }
    private fun emulator() = Build.FINGERPRINT.contains("generic") || Build.MODEL.contains("sdk", true) || Build.HARDWARE.contains("ranchu")
    private fun floats(raw: ByteArray): FloatArray { require(raw.size % 4 == 0); val b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer(); return FloatArray(b.remaining()).also { b.get(it) } }
    private fun bytes(a: FloatArray) = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { asFloatBuffer().put(a) }.array()
    private fun hash(raw: ByteArray) = MessageDigest.getInstance("SHA-256").digest(raw).joinToString("") { "%02x".format(it.toInt() and 255) }
}
