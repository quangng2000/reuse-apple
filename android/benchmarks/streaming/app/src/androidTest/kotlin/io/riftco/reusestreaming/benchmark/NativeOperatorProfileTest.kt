package io.riftco.reusestreaming.benchmark

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.ceil

/** Opt-in fixed-V1 graph diagnostic; never selected by the public streaming API. */
@RunWith(AndroidJUnit4::class)
class NativeOperatorProfileTest {
    @Test(timeout = 150_000L)
    fun profileFrozenEightLayerGraph() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val args = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val runId = args.getString("profileRunId") ?: "v1-profile-${System.currentTimeMillis()}"
        require(runId.matches(Regex("[A-Za-z0-9_.-]{1,80}")))
        val threads = args.getString("profileThreads")?.toIntOrNull() ?: 4
        require(threads in 1..8)
        val requirePhysical = when (args.getString("profileRequirePhysical")) {
            null, "true" -> true
            "false" -> false
            else -> error("profileRequirePhysical must be true or false.")
        }
        val emulator = Build.FINGERPRINT.contains("generic") || Build.MODEL.contains("sdk", true) || Build.HARDWARE.contains("ranchu")
        require(!requirePhysical || !emulator) { "Physical device required." }
        require(context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0) { "Nondebuggable target required." }
        require(Build.SUPPORTED_ABIS.contains("arm64-v8a"))
        val output = File(checkNotNull(context.getExternalFilesDir(null)), "firebase-profile").apply { check(mkdirs() || isDirectory) }
        val work = File(context.cacheDir, runId).apply { check(mkdirs() || isDirectory) }
        val summary = JSONObject().put("schemaVersion", 1).put("runId", runId).put("success", false)
            .put("scope", "Fixed pinned V1 ONNX spectral diagnostic; excludes SDK framing/FFT/synthesis and public push; profiler changes latency")
            .put("precision", "FP32").put("executionProvider", "CPUExecutionProvider").put("threads", threads)
            .put("device", JSONObject().put("model", Build.MODEL).put("fingerprint", Build.FINGERPRINT)
                .put("sdkInt", Build.VERSION.SDK_INT).put("emulatorReported", emulator).put("targetDebuggable", false))
        val started = SystemClock.elapsedRealtime()
        val deadline = started + 145_000L
        val wake = (context.getSystemService(Context.POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ReuseStreaming:NativeProfile").apply {
                setReferenceCounted(false)
                acquire(150_000L)
            }
        var foregroundHost: android.app.Activity? = null
        try {
            foregroundHost = instrumentation.startActivitySync(Intent()
                .setClassName(context.packageName, "${context.packageName}.MainActivity")
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                .putExtra("profileHostOnly", true))
            instrumentation.waitForIdleSync()
            fun asset(name: String, expected: String): ByteArray = context.assets.open(name).use { it.readBytes() }
                .also { check(sha256(it) == expected) { "Frozen V1 asset SHA256 mismatch: $name" } }
            val modelBytes = asset("native/reuse-streaming-l8-native-frequency-v1.onnx", MODEL_SHA)
            check(modelBytes.size == 11_590_965)
            val model = File(work, "model-v1.onnx").apply { writeBytes(modelBytes) }
            val manifestBytes = asset("native/reuse-streaming-l8-native-frequency-v1.manifest.json", MANIFEST_SHA)
            val manifest = JSONObject(manifestBytes.toString(Charsets.UTF_8))
            check(manifest.getString("modelSha256") == MODEL_SHA && manifest.getString("precision") == "float32" && manifest.getInt("layers") == 8)
            check(manifest.getString("checkpointSha256") == CHECKPOINT_SHA && manifest.getLong("stateBytes") == 41_575_984L)
            val library = File(context.applicationInfo.nativeLibraryDir, "libreuse_streaming_scan.so")
            check(library.length() == 1_114_624L && sha256(library.readBytes()) == LIBRARY_SHA) { "Frozen V1 library SHA256 mismatch." }
            val libraryReceipt = JSONObject(asset("native/libreuse_streaming_scan.so.json", LIBRARY_RECEIPT_SHA).toString(Charsets.UTF_8))
            check(libraryReceipt.getString("sha256") == LIBRARY_SHA && libraryReceipt.getString("sourceSha256") == SOURCE_SHA)
            val magnitudeBytes = asset("parity/l8/spectral-magnitude.f32", MAGNITUDE_SHA)
            val phaseBytes = asset("parity/l8/spectral-phase.f32", PHASE_SHA)
            fun frames(bytes: ByteArray): Array<FloatArray> {
                check(bytes.size == 8 * 321 * 4)
                val view = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                return Array(8) { FloatArray(321).also { values -> view.get(values); check(values.all(Float::isFinite)) } }
            }
            val magnitudes = frames(magnitudeBytes)
            val phases = frames(phaseBytes)
            summary.put("modelSha256", MODEL_SHA).put("manifestSha256", MANIFEST_SHA).put("nativeLibrarySha256", LIBRARY_SHA)
                .put("nativeSourceSha256", SOURCE_SHA).put("stateBytesPerBank", manifest.getLong("stateBytes"))
                .put("input", JSONObject().put("scope", "Eight pinned synthetic spectral fixture frames repeated with carried state; not live audio")
                    .put("magnitudeSha256", MAGNITUDE_SHA).put("phaseSha256", PHASE_SHA).put("frames", 8))
            val environment = OrtEnvironment.getEnvironment()
            check(environment.version == "1.23.2")
            summary.put("onnxRuntimeVersion", environment.version)
            // Reflect the exact SDK JNI finite checker, preserving all state/output validation.
            // Reflection overhead is included in this separately timed diagnostic.
            val finite = Class.forName("io.riftco.reusestreaming.NativeStreamingBuffers")
                .getDeclaredMethod("allFinite", Array<ByteBuffer>::class.java, IntArray::class.java)
            fun phase(profile: Boolean): Pair<JSONObject, File?> {
                val options = OrtSession.SessionOptions()
                var session: OrtSession? = null
                var io: PinnedIO? = null
                var run: OrtSession.RunOptions? = null
                var trace: File? = null
                try {
                    options.setIntraOpNumThreads(threads)
                    options.setInterOpNumThreads(1)
                    options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                    options.registerCustomOpLibrary(library.absolutePath)
                    if (profile) options.enableProfiling(File(work, "ort-v1-trace").absolutePath)
                    session = environment.createSession(model.absolutePath, options)
                    val activeSession = checkNotNull(session)
                    check(activeSession.metadata.customMetadata["execution_variant"] == "native-frequency-scan-v1")
                    io = PinnedIO(environment, activeSession, manifest)
                    val activeIO = checkNotNull(io)
                    run = OrtSession.RunOptions()
                    val activeRun = checkNotNull(run)
                    val warmup = if (profile) 2 else 8
                    val measured = if (profile) 3 else 20
                    val ortMs = DoubleArray(measured)
                    val finiteMs = DoubleArray(measured)
                    val spectrumCopyMs = DoubleArray(measured)
                    for (hop in 0 until warmup + measured) {
                        check(SystemClock.elapsedRealtime() < deadline && !Thread.currentThread().isInterrupted) { "Profile work deadline reached." }
                        activeIO.writeSpectrum(magnitudes[hop % 8], phases[hop % 8])
                        val ortStarted = SystemClock.elapsedRealtimeNanos()
                        activeSession.run(activeIO.inputs, emptySet(), activeIO.outputs, activeRun).use {
                            check(it.size() == 33)
                        }
                        val elapsedOrt = elapsedMs(ortStarted)
                        val finiteStarted = SystemClock.elapsedRealtimeNanos()
                        val isFinite = finite.invoke(null, activeIO.nextBytes, activeIO.outputCounts) as Boolean
                        val elapsedFinite = elapsedMs(finiteStarted)
                        check(isFinite) { "Nonfinite V1 output/state." }
                        val copyStarted = SystemClock.elapsedRealtimeNanos()
                        val copied = activeIO.copySpectrum()
                        val elapsedCopy = elapsedMs(copyStarted)
                        check(copied.all(Float::isFinite))
                        activeIO.commit()
                        if (hop >= warmup) {
                            ortMs[hop - warmup] = elapsedOrt
                            finiteMs[hop - warmup] = elapsedFinite
                            spectrumCopyMs[hop - warmup] = elapsedCopy
                        }
                    }
                    if (profile) trace = File(activeSession.endProfiling())
                    return JSONObject().put("warmupHops", warmup).put("measuredHops", measured)
                        .put("rawOrtRunMs", JSONArray(ortMs.toList())).put("ortRunMs", stats(ortMs))
                        .put("rawNativeFiniteCheckMs", JSONArray(finiteMs.toList())).put("nativeFiniteCheckMs", stats(finiteMs))
                        .put("finiteCheckScope", "Exact SDK JNI bitwise finite scan of all 33 output tensors, including reflection invocation overhead")
                        .put("finiteCheckedBytesPerHop", 41_575_984L + 642 * 4)
                        .put("rawSpectrumOutputCopyMs", JSONArray(spectrumCopyMs.toList())).put("spectrumOutputCopyMs", stats(spectrumCopyMs))
                        .put("spectrumCopyBytesPerHop", 642 * 4) to trace
                } finally {
                    // Registered custom code remains loaded through session/tensor destruction.
                    try { session?.close() } finally {
                        try { io?.close() } finally {
                            try { run?.close() } finally { options.close() }
                        }
                    }
                }
            }
            val unprofiled = phase(false)
            summary.put("unprofiled", unprofiled.first)
            val profiled = phase(true)
            summary.put("profiled", profiled.first)
            val traceSource = checkNotNull(profiled.second)
            check(traceSource.isFile && traceSource.length() in 1..60_000_000L)
            // Both sessions/options are closed before trace parsing, serialization and export.
            val traceBytes = traceSource.readBytes()
            val traceOutput = File(output, "$runId-ort-trace.json").apply { writeBytes(traceBytes) }
            summary.put("trace", JSONObject().put("file", traceOutput.absolutePath).put("bytes", traceBytes.size).put("sha256", sha256(traceBytes)))
                .put("profileAggregation", aggregateTrace(traceBytes))
                .put("profileOverheadRatio", profiled.first.getJSONObject("ortRunMs").getDouble("mean") /
                    unprofiled.first.getJSONObject("ortRunMs").getDouble("mean"))
                .put("profileCaution", "Trace operator durations include profiling overhead; use unprofiled run times for latency and do not equate diagnostic graph time with public SDK push.")
                .put("success", true)
        } catch (failure: Throwable) {
            summary.put("error", JSONObject().put("type", failure.javaClass.name).put("message", failure.message ?: "No message"))
            throw failure
        } finally {
            if (wake.isHeld) wake.release()
            foregroundHost?.let { activity -> instrumentation.runOnMainSync { activity.finish() } }
            summary.put("elapsedMs", SystemClock.elapsedRealtime() - started)
            val bytes = summary.toString(2).toByteArray(Charsets.UTF_8)
            val file = File(output, "$runId-profile-summary.json").apply { writeBytes(bytes) }
            File(output, "$runId-profile-summary.sha256").writeText(sha256(bytes) + "  " + file.name + "\n")
            Log.i("REUSE_PROFILE", "SUMMARY ${file.absolutePath} sha256=${sha256(bytes)} success=${summary.optBoolean("success")}")
        }
    }

    private class Tensor(environment: OrtEnvironment, val shape: LongArray) : AutoCloseable {
        val elements = shape.fold(1L) { product, dimension -> Math.multiplyExact(product, dimension) }
            .also { require(it in 1L..20_000_000L) }.toInt()
        val bytes: ByteBuffer = ByteBuffer.allocateDirect(elements * 4).order(ByteOrder.nativeOrder())
        val values = bytes.asFloatBuffer()
        val value = OnnxTensor.createTensor(environment, values, shape)
        init { check(!value.ownsBuffer() && value.bufferRef.isPresent) }
        fun write(input: FloatArray) { check(input.size == elements); values.position(0); values.put(input); values.position(0) }
        fun copyTo(output: FloatArray, offset: Int) { values.position(0); values.get(output, offset, elements); values.position(0) }
        override fun close() { value.close() }
    }

    private class PinnedIO(private val environment: OrtEnvironment, session: OrtSession, manifest: JSONObject) : AutoCloseable {
        private val owned = mutableListOf<Tensor>()
        private fun tensor(shape: LongArray) = Tensor(environment, shape).also { owned.add(it) }
        private val magnitude = tensor(longArrayOf(1, 321, 1))
        private val phase = tensor(longArrayOf(1, 321, 1))
        private val magnitudeOut = tensor(longArrayOf(1, 321, 1))
        private val phaseOut = tensor(longArrayOf(1, 321, 1))
        private val specs = manifest.getJSONArray("states").let { states -> (0 until states.length()).map { states.getJSONObject(it) } }
        private val banks = Array(2) { specs.map { spec ->
            val shape = spec.getJSONArray("shape")
            tensor(LongArray(shape.length()) { shape.getLong(it) })
        } }
        private val feeds = Array(2) { bank -> linkedMapOf("magnitude" to magnitude.value, "phase" to phase.value).apply {
            specs.forEachIndexed { index, spec -> put(spec.getString("inputName"), banks[bank][index].value) }
        } }
        private val outs = Array(2) { bank -> linkedMapOf("enhanced_magnitude" to magnitudeOut.value, "enhanced_phase" to phaseOut.value).apply {
            specs.forEachIndexed { index, spec -> put(spec.getString("outputName"), banks[1 - bank][index].value) }
        } }
        private val bytes = Array(2) { bank -> (listOf(magnitudeOut.bytes, phaseOut.bytes) + banks[1 - bank].map { it.bytes }).toTypedArray() }
        val outputCounts = intArrayOf(321, 321) + banks[0].map { it.elements }.toIntArray()
        private var committed = 0
        init {
            check(specs.size == 31 && session.inputNames == feeds[0].keys && session.outputNames == outs[0].keys)
            for ((name, value) in feeds[0]) {
                val info = session.inputInfo.getValue(name).info as TensorInfo
                check(info.type == OnnxJavaType.FLOAT && info.shape.contentEquals((value.info as TensorInfo).shape))
            }
            for ((name, value) in outs[0]) {
                val info = session.outputInfo.getValue(name).info as TensorInfo
                check(info.type == OnnxJavaType.FLOAT && info.shape.contentEquals((value.info as TensorInfo).shape))
            }
        }
        val inputs get() = feeds[committed]
        val outputs get() = outs[committed]
        val nextBytes get() = bytes[committed]
        fun writeSpectrum(mag: FloatArray, ph: FloatArray) { magnitude.write(mag); phase.write(ph) }
        fun copySpectrum() = FloatArray(642).also { magnitudeOut.copyTo(it, 0); phaseOut.copyTo(it, 321) }
        fun commit() { committed = 1 - committed }
        override fun close() { owned.forEach { it.close() } }
    }

    private fun aggregateTrace(raw: ByteArray): JSONObject {
        val events = JSONArray(raw.toString(Charsets.UTF_8))
        val runs = (0 until events.length()).map { events.getJSONObject(it) }
            .filter { it.optString("name") == "model_run" && it.optString("ph") == "X" }
        check(runs.size == 5)
        val steady = runs.drop(2)
        val groups = linkedMapOf<String, Pair<Int, Double>>()
        val nodes = linkedMapOf<String, Pair<Int, Double>>()
        for (index in 0 until events.length()) {
            val event = events.getJSONObject(index)
            if (event.optString("cat") != "Node" || event.optString("ph") != "X" || !event.optString("name").endsWith("_kernel_time")) continue
            val start = event.getDouble("ts")
            val duration = event.getDouble("dur")
            if (steady.none { start >= it.getDouble("ts") && start + duration <= it.getDouble("ts") + it.getDouble("dur") }) continue
            val operator = event.optJSONObject("args")?.optString("op_name") ?: "unknown"
            fun add(target: MutableMap<String, Pair<Int, Double>>, key: String) {
                val previous = target[key] ?: (0 to 0.0)
                target[key] = previous.first + 1 to previous.second + duration
            }
            add(groups, operator)
            add(nodes, event.getString("name"))
        }
        fun rows(data: Map<String, Pair<Int, Double>>, key: String) = JSONArray(data.entries.sortedByDescending { it.value.second }.map {
            JSONObject().put(key, it.key).put("calls", it.value.first).put("durationUs", it.value.second)
                .put("meanPerHopMs", it.value.second / steady.size / 1000.0)
        })
        return JSONObject().put("profiledSteadyHops", steady.size)
            .put("modelRunMeanMs", steady.sumOf { it.getDouble("dur") } / steady.size / 1000.0)
            .put("operators", rows(groups, "operator")).put("nodes", rows(nodes, "name"))
    }

    private fun stats(values: DoubleArray): JSONObject {
        val sorted = values.sortedArray()
        return JSONObject().put("mean", values.average()).put("p50", sorted[(ceil(.5 * sorted.size).toInt() - 1)])
            .put("p95", sorted[(ceil(.95 * sorted.size).toInt() - 1)]).put("min", sorted.first()).put("max", sorted.last())
    }
    private fun elapsedMs(started: Long) = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }

    companion object {
        private const val MODEL_SHA = "6b6cbd4117146f3137c1a0db71ecf04c23b8ba97bab7d973ece71c8739aa7756"
        private const val MANIFEST_SHA = "169b4600b34c649c86121e1ab39fd0b85d8f4eafa98e3891383743aacb594c6b"
        private const val LIBRARY_SHA = "d37c8919108be9ccc95d5297e88b31744736db6d60508372a77f4323fed38437"
        private const val LIBRARY_RECEIPT_SHA = "841a33a75ffc312f5ebd69ab8fb4efec535059c7529505f76a7bb379217f6118"
        private const val SOURCE_SHA = "1708573ce611abc9da7ab8453fbd51c7635f61dd8f5db77ee14788002f6ac030"
        private const val CHECKPOINT_SHA = "6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade"
        private const val MAGNITUDE_SHA = "02dd48f99a3a3f5dd2ff2298eb16a5cecdf848d2fefd72da39c243d8e7b05216"
        private const val PHASE_SHA = "308853c7d215cd95f04c7da271399f152779f02b6745412bcb97cfbbf02d1eb0"
    }
}
