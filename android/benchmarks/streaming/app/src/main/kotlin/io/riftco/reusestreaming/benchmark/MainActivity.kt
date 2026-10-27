package io.riftco.reusestreaming.benchmark

import android.app.Activity
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.Bundle
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import android.util.Base64
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.widget.TextView
import io.riftco.reusestreaming.StreamingCancellationToken
import io.riftco.reusestreaming.StreamingEnhancer
import io.riftco.reusestreaming.StreamingEnhancerConfiguration
import io.riftco.reusestreaming.StreamingModelManifest
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.concurrent.Executors
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.sin
import kotlin.math.sqrt

/** Private research harness for independently pinned FP32 streaming backends. */
class MainActivity : Activity() {
    private val worker = Executors.newSingleThreadExecutor { runnable -> Thread(runnable, "ReuseStreamingBenchmark") }
    @Volatile private var activeToken: StreamingCancellationToken? = null
    @Volatile private var finished = false
    @Volatile private var wakeLock: PowerManager.WakeLock? = null
    private lateinit var statusText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val runId = intent.getStringExtra("runId") ?: "emulator-${System.currentTimeMillis()}"
        Log.i(TAG, "RUN_START $runId")
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        statusText = TextView(this).apply {
            text = "RE-USE streaming benchmark\nPreparing…"
            textSize = 20f
            gravity = Gravity.CENTER
            setPadding(32, 32, 32, 32)
        }
        setContentView(statusText)
        if (intent.getBooleanExtra("profileHostOnly", false)) {
            // Keep the diagnostic in the same foreground app state as normal benchmarks.
            // The instrumentation test owns profiling and its bounded wake lock.
            statusText.text = "Eight-layer FP32 operator profile"
            finished = true
            return
        }
        window.decorView.postDelayed({
            if (!finished) {
                activeToken?.cancel()
                worker.shutdownNow()
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }, MAX_RUN_MS)
        worker.execute { execute(runId) }
    }

    override fun onDestroy() {
        if (!finished) activeToken?.cancel()
        worker.shutdownNow()
        releaseWakeLock()
        super.onDestroy()
    }

    private data class Case(val layers: Int, val threads: Int,
                            val backend: String = if (BuildConfig.NATIVE_FREQUENCY_SCAN) "frequency-v1" else "standard")
    private data class ModelPin(val bytes: Long, val sha256: String)
    private data class NativeBackend(val stem: String, val assetPrefix: String, val library: String,
                                     val factory: String, val pin: ModelPin)
    private data class Input(val initial: FloatArray, val hops: Array<FloatArray>, val sha256: String,
                             val totalSamples: Int, val exportSamples: FloatArray? = null)

    private fun execute(runId: String) {
        val started = SystemClock.elapsedRealtimeNanos()
        val deadline = started + MAX_RUN_MS * 1_000_000L
        val cases = JSONArray()
        val result = JSONObject().put("schemaVersion", 1).put("runId", runId)
            .put("success", false).put("status", "running").put("cases", cases)
            .put("sdkCoordinate", "io.github.quangng2000:reuse-streaming-android:0.1.3")
            .put("sdkSource", BuildConfig.STREAMING_SDK_SOURCE).put("variant", BuildConfig.STREAMING_VARIANT)
            .put("nativeFrequencyScan", BuildConfig.NATIVE_FREQUENCY_SCAN)
            .put("precision", "FP32").put("executionProvider", "CPUExecutionProvider")
            .put("onnxRuntimeVersion", "1.23.2").put("sampleRate", SAMPLE_RATE).put("hopSamples", HOP)
            .put("windowSamples", WINDOW).put("hopDeadlineMs", 20.0)
            .put("environmentScope", "In-process Android FP32 CPU processing; excludes network and audio-device latency")
        var currentCase: String? = null
        try {
            require(runId.matches(Regex("[A-Za-z0-9_.-]{1,80}"))) { "runId must contain 1–80 safe filename characters." }
            val debuggable = applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0
            val device = deviceInfo(debuggable)
            result.put("device", device)
                .put("environmentScope", if (device.getBoolean("emulatorReported"))
                    "Android emulator measurement; host-dependent, not physical-phone latency" else
                    "Physical Android device CPU processing; excludes network and audio-device latency")
            check(!debuggable) { "The benchmark target must be nondebuggable." }
            val timedFrames = intent.getIntExtra("timedFrames", 300)
            require(timedFrames in 32..1500) { "timedFrames must be between 32 and 1500." }
            val inputAsset = intent.getStringExtra("inputAsset")
            val captureOutput = intent.getBooleanExtra("captureOutput", false)
            val plan = plan()
            result.put("timedFramesPerCase", timedFrames).put("warmupHopsPerCase", WARMUP_HOPS)
                .put("plan", JSONArray(plan.map { JSONObject().put("layers", it.layers).put("threads", it.threads).put("backend", it.backend) }))
                .put("statistics", "Nearest-rank percentiles of individual steady push calls")
                .put("rtfDefinition", "Sum of steady push call times / timed input audio duration; excludes setup, warmup, checks, memory sampling and finish")
            val power = getSystemService(POWER_SERVICE) as PowerManager
            wakeLock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ReuseStreaming:Benchmark").apply {
                setReferenceCounted(false)
                acquire(MAX_RUN_MS)
            }
            val input = makeInput(timedFrames, inputAsset, captureOutput)
            result.put("inputSha256", input.sha256).put("qualityCapture", captureOutput)
            if (inputAsset == null) {
                result.put("syntheticInputSha256", input.sha256)
                    .put("inputDescription", "Continuous deterministic 173/317/911 Hz tones plus seeded xorshift noise; no recordings")
            } else {
                result.put("inputAsset", inputAsset)
                    .put("inputDescription", "Exact finite mono 16 kHz FP32 little-endian samples from a packaged validation asset")
            }
            for ((index, case) in plan.withIndex()) {
                checkDeadline(deadline)
                currentCase = "${index + 1}-l${case.layers}-t${case.threads}"
                progress("$currentCase starting (${index + 1}/${plan.size})")
                val measured = benchmarkCase(runId, currentCase, case, timedFrames, input, captureOutput, deadline)
                cases.put(measured)
                progress("$currentCase complete p50=${measured.getJSONObject("latencyMs").getDouble("p50")}ms")
            }
            result.put("success", true).put("status", "complete")
        } catch (failure: Throwable) {
            result.put("success", false).put("status", "failed").put("failedCaseId", currentCase ?: JSONObject.NULL)
                .put("error", JSONObject().put("type", failure.javaClass.name)
                    .put("message", failure.message?.take(2000) ?: "No error message")
                    .put("stackTrace", failure.stackTraceToString().take(8000)))
            Log.e(TAG, "FAILED ${failure.javaClass.simpleName}: ${failure.message}")
        } finally {
            activeToken = null
            releaseWakeLock()
            finished = true
            result.put("completedCases", cases.length())
                .put("elapsedMs", (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0)
            emitResult(runId, result)
            runOnUiThread {
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                statusText.text = if (result.optBoolean("success")) "Benchmark complete\n${cases.length()} cases saved" else
                    "Benchmark failed\n${result.optJSONObject("error")?.optString("message")}\n${cases.length()} completed cases saved"
            }
        }
    }

    private fun plan(): List<Case> {
        intent.getStringExtra("comparisonCases")?.let { selection ->
            require(BuildConfig.NATIVE_FREQUENCY_SCAN && selection.length in 1..300)
            val entries = selection.split('+')
            require(entries.size in 1..8)
            return entries.map { entry ->
                val fields = entry.split('@')
                require(fields.size == 2 && fields[0] in NATIVE_BACKENDS)
                val threads = fields[1].toIntOrNull() ?: error("Invalid comparison thread count.")
                require(threads in 1..8)
                Case(8, threads, fields[0])
            }
        }
        val primaryOnly = intent.getBooleanExtra("primaryOnly", false)
        val customLayers = intent.getStringExtra("layers")
        val customThreads = intent.getStringExtra("threads")
        fun numbers(value: String, allowed: Set<Int>, label: String): List<Int> {
            require(value.length in 1..100) { "Invalid $label list." }
            val entries = value.split(',').map { it.trim().toIntOrNull() ?: error("Invalid $label integer.") }
            require(entries.isNotEmpty() && entries.size <= allowed.size && entries.toSet().size == entries.size && entries.all { it in allowed }) {
                "Invalid or duplicate $label values."
            }
            return entries
        }
        if (!primaryOnly && customLayers == null && customThreads == null)
            return listOf(Case(8, 2), Case(4, 2), Case(8, 1), Case(8, 4), Case(12, 2), Case(8, 2))
        val layers = customLayers?.let { numbers(it, setOf(4, 8, 12), "layers") } ?: if (primaryOnly) listOf(8) else listOf(4, 8, 12)
        require(!primaryOnly || layers == listOf(8)) { "primaryOnly requires only layer 8." }
        val threads = customThreads?.let { numbers(it, (1..8).toSet(), "threads") } ?: listOf(2)
        val orderedLayers = listOf(8, 4, 12).filter { it in layers }
        return orderedLayers.flatMap { layer -> threads.map { Case(layer, it) } }
            .sortedBy { if (it == Case(8, 2)) 0 else 1 }
    }

    private fun benchmarkCase(runId: String, caseId: String, case: Case, timedFrames: Int, input: Input,
                              captureOutput: Boolean, deadline: Long): JSONObject {
        val native = if (case.backend == "standard") null else NATIVE_BACKENDS.getValue(case.backend)
        require(native == null || (BuildConfig.NATIVE_FREQUENCY_SCAN && case.layers == 8)) { "Native experiment supports exactly eight layers." }
        val pin = native?.pin ?: MODEL_PINS.getValue(case.layers)
        val folder = File(filesDir, "benchmark-models").apply { check(mkdirs() || isDirectory) }
        val stem = native?.stem ?: "reuse-streaming-l${case.layers}-fp32"
        val assetPrefix = native?.assetPrefix ?: "models"
        val modelFile = copyAsset("$assetPrefix/$stem.onnx", File(folder, "$stem.onnx"))
        val manifestFile = copyAsset("$assetPrefix/$stem.manifest.json", File(folder, "$stem.manifest.json"))
        val manifest = StreamingModelManifest.read(manifestFile)
        require(manifest.layers == case.layers && manifest.modelSha256 == pin.sha256 && manifest.modelBytes == pin.bytes)
        require(modelFile.length() == pin.bytes && sha256(modelFile.readBytes()) == pin.sha256) { "Pinned model hash/size mismatch." }
        val manifestJson = JSONObject(manifestFile.readText())
        val memory = JSONObject().put("beforeModel", memorySnapshot())
        val token = StreamingCancellationToken()
        activeToken = token
        val openStarted = SystemClock.elapsedRealtimeNanos()
        val configuration = StreamingEnhancerConfiguration(case.layers, case.threads)
        val enhancer = if (native != null) {
            val library = File(applicationInfo.nativeLibraryDir, native.library)
            val receipt = copyAsset("$assetPrefix/${native.library}.json", File(folder, "${native.library}.json"))
            try {
                val factory = Class.forName("io.riftco.reusestreaming.${native.factory}")
                factory.getMethod("create", File::class.java, File::class.java, File::class.java, File::class.java,
                    StreamingEnhancerConfiguration::class.java).invoke(null, modelFile, manifestFile, library, receipt, configuration) as StreamingEnhancer
            } catch (failure: java.lang.reflect.InvocationTargetException) {
                throw failure.targetException
            }
        } else StreamingEnhancer(modelFile, manifest, configuration)
        val modelOpenMs = elapsedMs(openStarted)
        var outcome: JSONObject? = null
        try {
            memory.put("afterModel", memorySnapshot())
            checkDeadline(deadline)
            progress("$caseId validating independent fixture")
            val fixture = validateFixture(enhancer, case.layers, manifest, token)
            enhancer.reset()
            progress("$caseId warm-up and steady benchmark")
            val warmupStarted = SystemClock.elapsedRealtimeNanos()
            enhancer.warmUp(token)
            val warmUpCallMs = elapsedMs(warmupStarted)
            val capturedOutput = if (captureOutput) FloatArray(input.totalSamples) else null
            var capturedSamples = 0
            val startupStarted = SystemClock.elapsedRealtimeNanos()
            val startupOutput = enhancer.push(input.initial, token)
            val startupCallMs = elapsedMs(startupStarted)
            checkAudio(startupOutput, HOP)
            val startupRenderMs = enhancer.lastFrameMilliseconds
            check(startupRenderMs.size == 2) { "Expected exactly two reflected startup frames." }
            var outputSamples = startupOutput.size.toLong()
            if (capturedOutput != null) {
                startupOutput.copyInto(capturedOutput, capturedSamples)
                capturedSamples += startupOutput.size
            }
            for (hop in 0 until WARMUP_HOPS) {
                checkDeadline(deadline)
                val output = enhancer.push(input.hops[hop], token)
                checkAudio(output, HOP)
                outputSamples += output.size
                if (capturedOutput != null) {
                    output.copyInto(capturedOutput, capturedSamples)
                    capturedSamples += output.size
                }
            }
            memory.put("afterWarmup", memorySnapshot())
            val gcBefore = gcStats()
            val rawPushMs = DoubleArray(timedFrames)
            val rawRenderMs = DoubleArray(timedFrames)
            var checksum = 0.0
            Log.i(TAG, "PROGRESS $caseId measuring $timedFrames steady hops")
            val loopStarted = SystemClock.elapsedRealtimeNanos()
            for (frame in 0 until timedFrames) {
                checkDeadline(deadline)
                val pushedAt = SystemClock.elapsedRealtimeNanos()
                val output = enhancer.push(input.hops[WARMUP_HOPS + frame], token)
                rawPushMs[frame] = elapsedMs(pushedAt)
                val render = enhancer.lastFrameMilliseconds
                check(render.size == 1) { "A steady 320-sample push must render exactly one frame." }
                rawRenderMs[frame] = render[0]
                checkAudio(output, HOP)
                outputSamples += output.size
                for (sample in output) checksum += sample.toDouble()
                if (capturedOutput != null) {
                    output.copyInto(capturedOutput, capturedSamples)
                    capturedSamples += output.size
                }
            }
            val loopWallMs = elapsedMs(loopStarted)
            val gcAfter = gcStats()
            val finishStarted = SystemClock.elapsedRealtimeNanos()
            val tail = enhancer.finish(token)
            val finishCallMs = elapsedMs(finishStarted)
            require(tail.all(Float::isFinite)) { "Nonfinite EOF samples." }
            outputSamples += tail.size
            check(outputSamples == input.totalSamples.toLong()) { "Exact output sample count was not preserved." }
            if (capturedOutput != null) {
                tail.copyInto(capturedOutput, capturedSamples)
                capturedSamples += tail.size
                check(capturedSamples == capturedOutput.size) { "Quality capture did not preserve every output sample." }
            }
            memory.put("afterFinish", memorySnapshot())
            val timedAudioMs = timedFrames * HOP * 1000.0 / SAMPLE_RATE
            val over20 = rawPushMs.count { it > 20.0 }
            outcome = JSONObject().put("caseId", caseId).put("success", true)
                .put("backend", case.backend)
                .put("layers", case.layers).put("threads", case.threads).put("timedFrames", timedFrames)
                .put("modelSha256", manifest.modelSha256).put("modelBytes", manifest.modelBytes)
                .put("logicalPersistentStateBytes", manifestJson.getLong("stateBytes"))
                .put("latencyMs", statistics(rawPushMs)).put("renderLatencyMs", statistics(rawRenderMs))
                .put("rawPushMs", JSONArray(rawPushMs.toList())).put("rawRenderMs", JSONArray(rawRenderMs.toList()))
                .put("over20msCount", over20).put("over20msPercent", over20 * 100.0 / timedFrames)
                .put("rtf", rawPushMs.sum() / timedAudioMs).put("loopWallRtf", loopWallMs / timedAudioMs)
                .put("timedAudioSeconds", timedAudioMs / 1000.0).put("loopWallMs", loopWallMs)
                .put("modelOpenMs", modelOpenMs).put("modelOpenIncludesHashVerification", true)
                .put("filesystemCacheState", "not controlled; model provisioning excluded")
                .put("warmUpCallMs", warmUpCallMs).put("warmupHops", WARMUP_HOPS)
                .put("startupCallMs", startupCallMs).put("startupRenderMs", JSONArray(startupRenderMs.toList()))
                .put("startupScope", "First 640-sample push after fixture validation, reset and warmUp; not cold app startup")
                .put("inputSamplesBeforeFirstOutput", WINDOW).put("nominalCaptureStartupMs", 40.0)
                .put("finishCallMs", finishCallMs).put("finishOutputSamples", tail.size)
                .put("inputSamples", input.totalSamples).put("outputSamples", outputSamples)
                .put("allOutputFinite", true).put("timedOutputChecksum", checksum)
                .put("qualityCapture", captureOutput)
                .put("fixture", fixture).put("memory", memory)
                .put("gcStatsBeforeTimed", gcBefore).put("gcStatsAfterTimed", gcAfter)
            if (capturedOutput != null) {
                outcome.put("export", exportQualityAudio(runId, caseId, input, capturedOutput))
            }
        } finally {
            try { enhancer.close() } finally { activeToken = null }
        }
        memory.put("afterClose", memorySnapshot())
        return checkNotNull(outcome)
    }

    private fun validateFixture(enhancer: StreamingEnhancer, layers: Int, manifest: StreamingModelManifest,
                                token: StreamingCancellationToken): JSONObject {
        val prefix = "parity/l$layers"
        val receipt = JSONObject(assets.open("$prefix/parity.json").bufferedReader().use { it.readText() })
        require(receipt.getBoolean("passed") && receipt.getInt("layers") == layers && receipt.getString("dtype") == "float32" &&
            receipt.getString("modelSha256") == MODEL_PINS.getValue(layers).sha256 &&
            receipt.getString("checkpointSha256") == StreamingModelManifest.CHECKPOINT_SHA256)
        if (!BuildConfig.NATIVE_FREQUENCY_SCAN)
            require(manifest.modelSha256 == MODEL_PINS.getValue(layers).sha256)
        val entries = receipt.getJSONArray("fixtures")
        val candidates = (0 until entries.length()).map { entries.getJSONObject(it) }
        val selected = if (layers == 12) candidates.minBy { it.getInt("samples") } else candidates.maxBy { it.getInt("samples") }
        require(selected.getInt("samples") == if (layers == 12) 640 else 16_000) { "Missing required independent fixture length." }
        fun readFloats(name: String): FloatArray {
            require(name.matches(Regex("[A-Za-z0-9_.-]+\\.f32")))
            val data = assets.open("$prefix/$name").use { it.readBytes() }
            val record = receipt.getJSONObject("files").getJSONObject(name)
            require(record.getString("file") == name && data.size.toLong() == record.getLong("bytes") &&
                sha256(data) == record.getString("sha256") && data.size % 4 == 0) { "Fixture hash/size mismatch." }
            val buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            return FloatArray(buffer.remaining()).also { buffer.get(it); require(it.all(Float::isFinite)) }
        }
        val input = readFloats(selected.getString("inputFile"))
        val expected = readFloats(selected.getString("outputFile"))
        require(input.size == selected.getInt("samples") && expected.size == input.size)
        enhancer.reset()
        val actual = enhancer.push(input, token) + enhancer.finish(token)
        require(actual.size == expected.size && actual.all(Float::isFinite))
        val maxError = actual.indices.maxOf { abs(actual[it] - expected[it]) }
        val rms = sqrt(actual.indices.sumOf { (actual[it] - expected[it]).toDouble().let { e -> e * e } } / actual.size)
        require(maxError <= 3e-4f) { "Independent waveform error $maxError exceeds 0.0003." }
        val resetInput = input.copyOf(WINDOW)
        enhancer.reset()
        val first = enhancer.push(resetInput, token) + enhancer.finish(token)
        enhancer.reset()
        val second = enhancer.push(resetInput, token) + enhancer.finish(token)
        require(first.size == WINDOW && second.size == WINDOW && first.indices.all { first[it].toRawBits() == second[it].toRawBits() }) { "Reset failed bitwise repeat check." }
        enhancer.reset()
        return JSONObject().put("samples", input.size).put("inputFile", selected.getString("inputFile"))
            .put("outputFile", selected.getString("outputFile")).put("maximumAbsoluteError", maxError.toDouble())
            .put("rmsError", rms).put("absoluteTolerance", 0.0003).put("passed", true).put("resetBitwiseRepeatPassed", true)
    }

    private fun copyAsset(path: String, output: File): File {
        assets.open(path).use { input -> output.outputStream().use { input.copyTo(it) } }
        return output
    }

    private fun makeInput(timedFrames: Int, inputAsset: String?, captureOutput: Boolean): Input {
        val count = WINDOW + (WARMUP_HOPS + timedFrames) * HOP
        var noise = 0x1234ABCD
        val samples = if (inputAsset == null) {
            FloatArray(count) { index ->
                noise = noise xor (noise shl 13)
                noise = noise xor (noise ushr 17)
                noise = noise xor (noise shl 5)
                val random = (noise ushr 8) / 8_388_607.5 - 1.0
                (0.12 * sin(2 * PI * 173 * index / SAMPLE_RATE) + 0.07 * sin(2 * PI * 317 * index / SAMPLE_RATE) +
                    0.025 * sin(2 * PI * 911 * index / SAMPLE_RATE) + 0.025 * random).toFloat()
            }
        } else {
            require(inputAsset.length in 1..200 && inputAsset.endsWith(".f32") &&
                inputAsset.split('/').all { part ->
                    part != "." && part != ".." && part.matches(Regex("[A-Za-z0-9_.-]+"))
                }) { "inputAsset must be a safe relative APK asset path ending in .f32." }
            val data = assets.open(inputAsset).use { stream ->
                val expected = ByteArray(count * 4)
                var offset = 0
                while (offset < expected.size) {
                    val read = stream.read(expected, offset, expected.size - offset)
                    require(read > 0) { "Validation input is shorter than the exact required sample count." }
                    offset += read
                }
                require(stream.read() == -1) { "Validation input is longer than the exact required sample count." }
                expected
            }
            val floats = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            FloatArray(count).also { floats.get(it); require(it.all(Float::isFinite)) { "Validation input contains nonfinite samples." } }
        }
        val bytes = ByteBuffer.allocate(samples.size * 4).order(ByteOrder.LITTLE_ENDIAN).apply { asFloatBuffer().put(samples) }.array()
        return Input(samples.copyOf(WINDOW), Array(WARMUP_HOPS + timedFrames) { index ->
            samples.copyOfRange(WINDOW + index * HOP, WINDOW + (index + 1) * HOP)
        }, sha256(bytes), samples.size, if (captureOutput) samples else null)
    }

    private fun exportQualityAudio(runId: String, caseId: String, input: Input, output: FloatArray): JSONObject {
        val folder = File(checkNotNull(getExternalFilesDir(null)), "benchmark-export").apply { check(mkdirs() || isDirectory) }
        fun write(kind: String, samples: FloatArray): JSONObject {
            val bytes = ByteBuffer.allocate(samples.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                .apply { asFloatBuffer().put(samples) }.array()
            val file = File(folder, "$runId-$caseId-$kind.f32")
            file.writeBytes(bytes)
            val hash = sha256(bytes)
            if (kind == "input") check(hash == input.sha256) { "Exported input differs from the benchmark input." }
            return JSONObject().put("file", file.absolutePath).put("sha256", hash)
                .put("samples", samples.size).put("bytes", bytes.size)
        }
        return JSONObject().put("dtype", "float32-le").put("sampleRate", SAMPLE_RATE)
            .put("input", write("input", checkNotNull(input.exportSamples)))
            .put("output", write("output", output))
    }

    private fun checkAudio(output: FloatArray, expectedCount: Int) {
        require(output.size == expectedCount && output.all(Float::isFinite)) { "Invalid streaming output samples." }
    }

    private fun statistics(values: DoubleArray): JSONObject {
        require(values.isNotEmpty() && values.all { it.isFinite() && it >= 0 })
        val sorted = values.sortedArray()
        fun percentile(value: Double): Double = sorted[(ceil(value * sorted.size).toInt() - 1).coerceIn(0, sorted.lastIndex)]
        return JSONObject().put("mean", values.average()).put("min", sorted.first()).put("p50", percentile(0.5))
            .put("p95", percentile(0.95)).put("p99", percentile(0.99)).put("max", sorted.last())
    }

    private fun memorySnapshot(): JSONObject {
        val runtime = Runtime.getRuntime()
        val info = Debug.MemoryInfo()
        Debug.getMemoryInfo(info)
        return JSONObject().put("sampleElapsedRealtimeNs", SystemClock.elapsedRealtimeNanos())
            .put("pssBytes", info.totalPss.toLong() * 1024).put("nativePssBytes", info.nativePss.toLong() * 1024)
            .put("dalvikPssBytes", info.dalvikPss.toLong() * 1024).put("privateDirtyBytes", info.totalPrivateDirty.toLong() * 1024)
            .put("javaUsedBytes", runtime.totalMemory() - runtime.freeMemory()).put("javaHeapBytes", runtime.totalMemory())
            .put("javaMaxHeapBytes", runtime.maxMemory()).put("nativeAllocatedBytes", Debug.getNativeHeapAllocatedSize())
            .put("gcRuntimeStats", gcStats()).put("scope", "Boundary snapshot, not peak memory")
    }

    private fun gcStats(): JSONObject = JSONObject(Debug.getRuntimeStats().filterKeys { it.startsWith("art.gc.") })

    private fun deviceInfo(debuggable: Boolean): JSONObject = JSONObject()
        .put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL).put("device", Build.DEVICE)
        .put("fingerprint", Build.FINGERPRINT).put("sdkInt", Build.VERSION.SDK_INT).put("androidRelease", Build.VERSION.RELEASE)
        .put("abis", JSONArray(Build.SUPPORTED_ABIS.toList())).put("availableProcessors", Runtime.getRuntime().availableProcessors())
        .put("targetDebuggable", debuggable).put("packageName", packageName).put("appVersion", BuildConfig.VERSION_NAME)
        .put("emulatorReported", Build.FINGERPRINT.contains("generic") || Build.MODEL.contains("sdk", ignoreCase = true) || Build.HARDWARE.contains("ranchu"))

    private fun checkDeadline(deadline: Long) {
        check(!Thread.currentThread().isInterrupted && SystemClock.elapsedRealtimeNanos() < deadline) { "Benchmark interrupted or ten-minute runtime limit reached." }
    }

    private fun elapsedMs(started: Long): Double = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0
    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) }

    private fun progress(message: String) {
        Log.i(TAG, "PROGRESS $message")
        runOnUiThread { statusText.text = "RE-USE streaming benchmark\n$message" }
    }

    private fun releaseWakeLock() {
        try { wakeLock?.let { if (it.isHeld) it.release() } } catch (_: RuntimeException) { }
    }

    private fun emitResult(runId: String, result: JSONObject) {
        val output = File(filesDir, "result-${runId.replace(Regex("[^A-Za-z0-9_.-]"), "_").take(80)}.json")
        try {
            output.writeText(result.toString(), Charsets.UTF_8)
        } catch (failure: Throwable) {
            result.put("success", false).put("status", "failed").put("persistenceError", failure.javaClass.name + ": " + failure.message)
        }
        val bytes = result.toString().toByteArray(Charsets.UTF_8)
        val encoded = Base64.encodeToString(bytes, Base64.NO_WRAP)
        val chunks = encoded.chunked(3000)
        for ((index, chunk) in chunks.withIndex()) Log.i(TAG, "RESULT_CHUNK ${index + 1}/${chunks.size}:$chunk")
        Log.i(TAG, "RESULT_SHA256:${sha256(bytes)}")
    }

    companion object {
        private const val TAG = "REUSE_BENCH"
        private const val SAMPLE_RATE = 16_000
        private const val HOP = 320
        private const val WINDOW = 640
        private const val WARMUP_HOPS = 32
        private const val MAX_RUN_MS = 10 * 60 * 1000L
        private val NATIVE_MODEL_PIN = ModelPin(11_590_965, "6b6cbd4117146f3137c1a0db71ecf04c23b8ba97bab7d973ece71c8739aa7756")
        private val NATIVE_BACKENDS = mapOf(
            "frequency-v1" to NativeBackend("reuse-streaming-l8-native-frequency-v1", "native", "libreuse_streaming_scan.so",
                "ExperimentalNativeStreaming", NATIVE_MODEL_PIN),
            "bulk-v2" to NativeBackend("reuse-streaming-l8-native-frequency-temporal-v2", "native-v2-bulk", "libreuse_streaming_native_v2.so",
                "ExperimentalNativeStreamingV2", ModelPin(11_590_141, "1fbd91be9c90568c58ab40e9ce70e07c54dafc599fffb694a84c1f4d3905b11d")),
            "inline-v2" to NativeBackend("reuse-streaming-l8-native-inline-v2", "native-v2-inline", "libreuse_streaming_inline_v2.so",
                "ExperimentalNativeStreamingInlineV2", ModelPin(11_584_702, "55d910c30cd7144187e5f358dca6700ae6e7e1051eb8f7758d454f68e9144c11")),
            "stable-v2" to NativeBackend("reuse-streaming-l8-native-stable-v2", "native-v2-stable", "libreuse_streaming_stable_v2.so",
                "ExperimentalNativeStreamingStableV2", ModelPin(11_586_326, "db29c725c70608920e600787fcda625e98429cfbae2c2d09f1dc73726825319a"))
        )
        private val MODEL_PINS = mapOf(
            4 to ModelPin(8_217_659, "77d54456b8e87af87624f1989530b3c41d18a6cff65ccf32ab421a4b3b35829f"),
            8 to ModelPin(11_615_514, "18db8d32b638702909de484d2a9cffc97ed09e6454c4caed3f0008fb0aab53e8"),
            12 to ModelPin(15_014_464, "425fe4bf0c27f05505214bd565886bf266f52b21425fb78f1e70e81fb5caed3a"),
        )
    }
}
