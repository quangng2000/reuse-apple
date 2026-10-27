package io.riftco.reusestreaming.benchmark

import android.app.Activity
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.Process
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
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.sqrt

/** One final, bounded physical execution. Numerical gates precede the symmetric ABBA timings. */
@RunWith(AndroidJUnit4::class)
class VulkanFinalBudgetBenchmarkTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private lateinit var root: File
    private lateinit var artifacts: File
    private lateinit var runId: String
    private var deadline = 0L

    @Test(timeout = 55_000L)
    fun validateAndCompareWithinOneMinute() {
        val started = SystemClock.elapsedRealtime()
        deadline = started + 45_000L
        val args = InstrumentationRegistry.getArguments()
        runId = args.getString("vulkanRunId") ?: "vulkan-final-${System.currentTimeMillis()}"
        require(runId.matches(Regex("[A-Za-z0-9_.-]{1,80}")))
        val physical = args.getString("vulkanRequirePhysical") != "false"
        val frames = args.getString("vulkanFrames")?.toInt() ?: 160
        require(frames in 32..160)
        artifacts = File(checkNotNull(context.getExternalFilesDir(null)), "firebase-vulkan-v3").apply { mkdirs() }
        root = File(context.filesDir, "vulkan-validation-v3").apply { mkdirs() }
        val completed = AtomicBoolean(false)
        // The native fence can wait longer than the work budget. This independent process
        // watchdog ends even that path before the provider's fixed 60-second limit.
        val watchdog = Thread({
            try {
                while (!completed.get()) {
                    val remaining = started + 50_000L - SystemClock.elapsedRealtime()
                    if (remaining <= 0L) {
                        if (!completed.get()) {
                            // Do not perform file I/O before termination: a blocked write
                            // must not delay the monetary deadline. Missing final evidence fails verification.
                            Process.killProcess(Process.myPid())
                        }
                        break
                    }
                    Thread.sleep(minOf(remaining, 250L))
                }
            } catch (_: InterruptedException) { /* normal completion */ }
        }, "vulkan-budget-watchdog").apply { isDaemon = true; start() }
        val cases = JSONArray()
        val result = JSONObject().put("schemaVersion", 3).put("runId", runId).put("success", false)
            .put("scope", "Foreground synchronous 320-sample public streaming calls, including DSP, GPU completion and output readback; excludes audio-device and network latency")
            .put("layers", 8).put("precision", "FP32").put("hopSamples", 320).put("sampleRate", 16000)
            .put("timedFramesPerCase", frames).put("warmupHops", 16).put("caseOrder", "ABBA")
            .put("workDeadlineMs", 45000).put("processDeadlineMs", 50000).put("cases", cases)
            .put("device", JSONObject().put("model", Build.MODEL).put("manufacturer", Build.MANUFACTURER)
                .put("fingerprint", Build.FINGERPRINT).put("sdk", Build.VERSION.SDK_INT)
                .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList())).put("emulatorReported", emulator()))
        var activity: Activity? = null
        try {
            require(!physical || (!emulator() && Build.MODEL == "SM-S938U1" && Build.VERSION.SDK_INT == 35)) { "The approved physical Galaxy S25 Ultra is required." }
            require(context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) { "Release target required." }
            activity = instrumentation.startActivitySync(Intent().setClassName(context.packageName, "${context.packageName}.MainActivity")
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                .putExtra("runId", runId).putExtra("profileHostOnly", true))
            checkWork()
            copyTree("vulkan", File(root, "vulkan"))
            copyTree("vulkan-v2", File(root, "vulkan-v2"))
            val library = File(context.applicationInfo.nativeLibraryDir, "libreuse_streaming_vulkan.so")
            result.put("nativeLibrarySha256", hash(library.readBytes()))
            result.put("baselineBundleManifestSha256", hash(File(root, "vulkan-v2/manifest.json").readBytes()))
            result.put("bundleManifestSha256", hash(File(root, "vulkan/manifest.json").readBytes()))
            checkWork()
            ExperimentalVulkanStreaming.createTiled(File(root, "vulkan-v2"), library, StreamingEnhancerConfiguration(8, 1)).use { baseline ->
                ExperimentalVulkanStreaming.createTiled(File(root, "vulkan"), library, StreamingEnhancerConfiguration(8, 1)).use { candidate ->
                    result.put("baselineVulkan", checkDevice(baseline, false))
                    result.put("vulkan", checkDevice(candidate, false))
                    result.put("qualityBaseline", validateQuality(baseline, "baseline"))
                    result.put("quality", validateQuality(candidate, "optimized"))
                    val fullInput = asset("validation/vulkan-benchmark.f32")
                    require(hash(fullInput) == "e3150c5d89ff157b323627de4e5ede75a35be2e44088b284daa9d47268ccb9ca")
                    val input = floats(fullInput).copyOfRange(0, 640 + (16 + frames) * 320)
                    result.put("benchmarkInputSha256", hash(bytes(input)))
                    export("benchmark-input", input)
                    cases.put(measure(baseline, "vulkan-v2-first", "Vulkan", 0, input, frames))
                    cases.put(measure(candidate, "vulkan-v3-first", "Vulkan", 0, input, frames))
                    cases.put(measure(candidate, "vulkan-v3-second", "Vulkan", 0, input, frames))
                    cases.put(measure(baseline, "vulkan-v2-second", "Vulkan", 0, input, frames))
                    result.put("optionalProfile", JSONObject().put("performed", false).put("reason", "Insufficient remaining work budget or profiling disabled"))
                    if (args.getString("vulkanProfile") != "false" && deadline - SystemClock.elapsedRealtime() >= 10_000L) {
                        checkWork()
                        candidate.reset(); candidate.warmUp()
                        val normal = ArrayList<FloatArray>()
                        normal.add(candidate.push(input.copyOfRange(0, 640)))
                        repeat(4) { i -> normal.add(candidate.push(input.copyOfRange(640 + i * 320, 960 + i * 320))) }
                        ExperimentalVulkanStreaming.createProfiled(File(root, "vulkan"), library,
                            StreamingEnhancerConfiguration(8, 1), true).use { profiler ->
                            val device = checkDevice(profiler, true)
                            profiler.warmUp()
                            val first = profiler.push(input.copyOfRange(0, 640))
                            require(first.size == normal[0].size && first.indices.all { first[it].toRawBits() == normal[0][it].toRawBits() })
                            val profiles = JSONArray()
                            repeat(4) { i ->
                                checkWork()
                                val out = profiler.push(input.copyOfRange(640 + i * 320, 960 + i * 320))
                                require(out.size == 320 && out.all(Float::isFinite))
                                require(out.indices.all { out[it].toRawBits() == normal[i + 1][it].toRawBits() })
                                profiles.put(JSONObject(checkNotNull(profiler.backendProfileJson)))
                            }
                            result.put("optionalProfile", JSONObject().put("performed", true).put("vulkan", device)
                                .put("profiles", profiles).put("frames", 4).put("profiledOutputBitwiseEqual", true)
                                .put("scope", "Diagnostic per-dispatch timestamps only; excluded from all performance cases"))
                        }
                    }
                }
            }
            checkWork()
            result.put("success", true)
        } catch (failure: Throwable) {
            result.put("error", JSONObject().put("type", failure.javaClass.name)
                .put("message", failure.message ?: "No message").put("trace", failure.stackTraceToString().take(12000)))
            throw failure
        } finally {
            try {
                result.put("elapsedMs", SystemClock.elapsedRealtime() - started)
                val output = result.toString(2).toByteArray()
                File(artifacts, "$runId-result.json").writeBytes(output)
                File(artifacts, "$runId-result.sha256").writeText(hash(output) + "\n")
                Log.i("REUSE_VULKAN", "RESULT runId=$runId success=${result.optBoolean("success")} sha256=${hash(output)}")
                activity?.let { a -> instrumentation.runOnMainSync { a.finish() } }
            } finally { completed.set(true); watchdog.interrupt() }
        }
    }

    private fun checkDevice(enhancer: StreamingEnhancer, profiled: Boolean): JSONObject {
        val device = JSONObject(checkNotNull(enhancer.backendDiagnosticsJson))
        require(device.getBoolean("hardwareGpu") && !device.getBoolean("softwareDeviceAllowed") &&
            device.getString("precision") == "float32" && device.getInt("stateCount") == 31 &&
            device.getBoolean("statePingPong") && device.getBoolean("gpuFiniteChecks") &&
            device.getBoolean("tiledMatmulPipeline") && device.getBoolean("tiledConvolutionPipeline") &&
            device.getBoolean("perDispatchProfiling") == profiled)
        return device
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
        for (length in listOf(640, 641, 959, 960, 1377)) {
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
        repeat(16) {
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
    private fun checkWork() = check(SystemClock.elapsedRealtime() < deadline && !Thread.currentThread().isInterrupted) { "Vulkan validation exceeded 45-second work limit." }
    private fun emulator() = Build.FINGERPRINT.contains("generic") || Build.MODEL.contains("sdk", true) || Build.HARDWARE.contains("ranchu")
    private fun floats(raw: ByteArray): FloatArray { require(raw.size % 4 == 0); val b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer(); return FloatArray(b.remaining()).also { b.get(it) } }
    private fun bytes(a: FloatArray) = ByteBuffer.allocate(a.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { asFloatBuffer().put(a) }.array()
    private fun hash(raw: ByteArray) = MessageDigest.getInstance("SHA-256").digest(raw).joinToString("") { "%02x".format(it.toInt() and 255) }
}
