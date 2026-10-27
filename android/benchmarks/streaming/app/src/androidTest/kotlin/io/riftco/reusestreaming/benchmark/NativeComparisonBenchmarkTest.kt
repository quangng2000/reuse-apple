package io.riftco.reusestreaming.benchmark

import android.app.Activity
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.Bundle
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

/** One foreground execution compares ordered, independently pinned backends on the same phone. */
@RunWith(AndroidJUnit4::class)
class NativeComparisonBenchmarkTest {
    private data class Case(val backend: String, val threads: Int)

    @Test(timeout = 230_000L)
    fun comparePinnedEightLayerBackends() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val started = SystemClock.elapsedRealtime()
        val deadline = started + 210_000L
        val runId = arguments.getString("comparisonRunId") ?: "comparison-${System.currentTimeMillis()}"
        require(runId.matches(Regex("[A-Za-z0-9_.-]{1,80}")))
        val selection = arguments.getString("comparisonCases") ?: DEFAULT_CASES
        require(selection.length in 1..300)
        val plan = selection.split('+').map { entry ->
            val match = requireNotNull(Regex("(frequency-v1|bulk-v2|inline-v2|stable-v2)@([1-8])").matchEntire(entry)) {
                "Invalid comparisonCases entry."
            }
            Case(match.groupValues[1], match.groupValues[2].toInt())
        }
        require(plan.size in 1..8)
        val frames = integer(arguments, "comparisonFrames", 300, 32..1500)
        val physical = boolean(arguments, "comparisonRequirePhysical", true)
        val emulator = Build.FINGERPRINT.contains("generic") || Build.MODEL.contains("sdk", true) || Build.HARDWARE.contains("ranchu")
        require(!physical || !emulator) { "Physical target required." }
        require(context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE == 0)
        require(Build.SUPPORTED_ABIS.contains("arm64-v8a"))
        val capture = boolean(arguments, "comparisonCaptureOutput", false)
        val asset = arguments.getString("comparisonInputAsset")
        if (asset != null) require(asset.matches(Regex("[A-Za-z0-9_./-]+")) && ".." !in asset.split('/'))
        val expectedInput = arguments.getString("comparisonExpectedInputSha256") ?: DEFAULT_INPUT_SHA.also {
            require(frames == 300 && asset == null) { "Supply an explicit expected input hash for custom frames/audio." }
        }
        require(expectedInput.matches(Regex("[a-f0-9]{64}")))
        val external = checkNotNull(context.getExternalFilesDir(null))
        val artifacts = File(external, "firebase-comparison").apply { check(mkdirs() || isDirectory) }
        val receiptFile = File(artifacts, "$runId-receipt.json")
        val receipt = JSONObject().put("schemaVersion", 1).put("runId", runId).put("success", false)
            .put("instrumentationClass", javaClass.name)
            .put("scope", "Same-phone in-app CPU FP32 streaming comparison; excludes cloud/install/network and audio-device latency")
            .put("requested", JSONObject().put("comparisonCases", selection).put("timedFrames", frames)
                .put("requirePhysical", physical).put("captureOutput", capture).put("expectedInputSha256", expectedInput))
        val resultFile = File(context.filesDir, "result-$runId.json")
        check(!resultFile.exists() || resultFile.delete()) { "Could not clear owned stale result." }
        var activity: Activity? = null
        fun checkWork() = check(SystemClock.elapsedRealtime() < deadline && !Thread.currentThread().isInterrupted) {
            "Comparison exceeded its 210-second work deadline."
        }
        try {
            val launch = Intent().setClassName(context.packageName, "${context.packageName}.MainActivity")
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                .putExtra("runId", runId).putExtra("timedFrames", frames).putExtra("comparisonCases", selection)
                .putExtra("captureOutput", capture)
            asset?.let { launch.putExtra("inputAsset", it) }
            activity = instrumentation.startActivitySync(launch)
            var raw: ByteArray? = null
            var parsed: JSONObject? = null
            // No tracing, shell/UI automation or memory sampling runs during timed inference.
            while (parsed == null) {
                checkWork()
                if (resultFile.isFile && resultFile.length() in 1..10_000_000L) {
                    val candidate = resultFile.readBytes()
                    val json = try { JSONObject(candidate.toString(Charsets.UTF_8)) } catch (_: org.json.JSONException) { null }
                    if (json != null && json.optString("runId") == runId && json.optString("status") in setOf("complete", "failed")) {
                        raw = candidate; parsed = json
                    }
                }
                if (parsed == null) SystemClock.sleep(1_000L)
            }
            val result = checkNotNull(parsed); val bytes = checkNotNull(raw)
            val saved = File(artifacts, "$runId-result.json").apply { writeBytes(bytes) }
            File(artifacts, "$runId-result.sha256").writeText(sha256(bytes) + "  " + saved.name + "\n")
            receipt.put("result", JSONObject().put("file", saved.absolutePath).put("bytes", bytes.size).put("sha256", sha256(bytes)))
            check(result.getBoolean("success") && result.getString("status") == "complete") { "Comparison app reported failure." }
            check(result.getInt("schemaVersion") == 1 && result.getString("variant") == "native-v2-comparison")
            check(result.getString("sdkCoordinate") == "io.github.quangng2000:reuse-streaming-android:0.1.2" &&
                result.getString("sdkSource") == "local" && result.getString("precision") == "FP32" &&
                result.getString("executionProvider") == "CPUExecutionProvider" && result.getString("onnxRuntimeVersion") == "1.23.2")
            check(result.getInt("sampleRate") == 16000 && result.getInt("hopSamples") == 320 && result.getInt("windowSamples") == 640)
            val device = result.getJSONObject("device")
            check(!device.getBoolean("targetDebuggable")) { "Target must be nondebuggable." }
            if (physical) check(!device.getBoolean("emulatorReported")) { "Physical target required." }
            val abis = device.getJSONArray("abis")
            check((0 until abis.length()).any { abis.getString(it) == "arm64-v8a" })
            check(result.optString("inputSha256", result.optString("syntheticInputSha256")) == expectedInput) { "Input hash mismatch." }
            check(result.getInt("timedFramesPerCase") == frames && result.getInt("completedCases") == plan.size)
            val cases = result.getJSONArray("cases"); val observedPlan = result.getJSONArray("plan")
            check(cases.length() == plan.size && observedPlan.length() == plan.size)
            val caseIds = mutableSetOf<String>(); val exports = JSONArray()
            for ((index, expected) in plan.withIndex()) {
                checkWork()
                val declared = observedPlan.getJSONObject(index); val case = cases.getJSONObject(index)
                for (row in listOf(declared, case)) check(row.getInt("layers") == 8 && row.getInt("threads") == expected.threads &&
                    row.getString("backend") == expected.backend) { "Comparison case order or identity mismatch." }
                val caseId = case.getString("caseId")
                check(caseId.matches(Regex("[A-Za-z0-9_.-]{1,100}")) && caseIds.add(caseId))
                check(case.getBoolean("success") && case.getInt("timedFrames") == frames && case.getBoolean("allOutputFinite"))
                check(case.getString("modelSha256") == MODEL_HASHES.getValue(expected.backend)) { "Untrusted comparison model." }
                val samples = case.getInt("inputSamples")
                check(samples >= 640 && samples == case.getInt("outputSamples"))
                if (asset == null) check(samples == 640 + (result.getInt("warmupHopsPerCase") + frames) * 320)
                val fixture = case.getJSONObject("fixture")
                check(fixture.getBoolean("passed") && fixture.getBoolean("resetBitwiseRepeatPassed") && fixture.getInt("samples") == 16000)
                check(fixture.getDouble("maximumAbsoluteError").let { it.isFinite() && it >= 0 && it <= .0003 })
                check(case.getDouble("timedOutputChecksum").isFinite() && case.getDouble("rtf").let { it.isFinite() && it > 0 })
                for (name in listOf("rawPushMs", "rawRenderMs")) {
                    val times = case.getJSONArray(name); check(times.length() == frames)
                    for (hop in 0 until frames) check(times.getDouble(hop).let { it.isFinite() && it >= 0 })
                }
                check(case.getBoolean("qualityCapture") == capture)
                if (capture) {
                    val exported = case.getJSONObject("export")
                    check(exported.getString("dtype") == "float32-le" && exported.getInt("sampleRate") == 16000)
                    for (kind in listOf("input", "output")) {
                        val record = exported.getJSONObject(kind); val source = File(record.getString("file"))
                        check(source.canonicalFile.parentFile == File(external, "benchmark-export").canonicalFile &&
                            source.name == "$runId-$caseId-$kind.f32")
                        val pcm = source.readBytes()
                        check(record.getInt("samples") == samples && record.getInt("bytes") == samples * 4 && pcm.size == samples * 4 &&
                            sha256(pcm) == record.getString("sha256"))
                        val floats = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                        while (floats.hasRemaining()) check(floats.get().isFinite())
                        if (kind == "input") check(sha256(pcm) == expectedInput)
                        val savedPcm = File(artifacts, source.name).apply { writeBytes(pcm) }
                        exports.put(JSONObject().put("file", savedPcm.absolutePath).put("bytes", pcm.size).put("sha256", sha256(pcm)))
                    }
                }
            }
            receipt.put("exports", exports).put("success", true)
        } catch (failure: Throwable) {
            receipt.put("error", JSONObject().put("type", failure.javaClass.name).put("message", failure.message ?: "No message"))
            throw failure
        } finally {
            try { activity?.let { launched -> instrumentation.runOnMainSync { launched.finish() } } } finally {
                receipt.put("elapsedMs", SystemClock.elapsedRealtime() - started)
                val bytes = receipt.toString(2).toByteArray(Charsets.UTF_8)
                receiptFile.writeBytes(bytes)
                File(artifacts, "$runId-receipt.sha256").writeText(sha256(bytes) + "  " + receiptFile.name + "\n")
                Log.i("REUSE_COMPARISON", "RECEIPT ${receiptFile.absolutePath} sha256=${sha256(bytes)} success=${receipt.optBoolean("success")}")
            }
        }
    }

    private fun integer(args: Bundle, name: String, default: Int, allowed: IntRange): Int {
        val value = args.getString(name)?.let { it.toIntOrNull() ?: error("Invalid $name.") } ?: default
        require(value in allowed); return value
    }
    private fun boolean(args: Bundle, name: String, default: Boolean): Boolean = args.getString(name)?.let {
        when (it) { "true" -> true; "false" -> false; else -> error("Invalid $name boolean.") }
    } ?: default
    private fun sha256(bytes: ByteArray) = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 255) }

    companion object {
        private const val DEFAULT_CASES = "frequency-v1@4+stable-v2@4+stable-v2@8+stable-v2@6+frequency-v1@4"
        private const val DEFAULT_INPUT_SHA = "e3150c5d89ff157b323627de4e5ede75a35be2e44088b284daa9d47268ccb9ca"
        private val MODEL_HASHES = mapOf(
            "frequency-v1" to "6b6cbd4117146f3137c1a0db71ecf04c23b8ba97bab7d973ece71c8739aa7756",
            "bulk-v2" to "1fbd91be9c90568c58ab40e9ce70e07c54dafc599fffb694a84c1f4d3905b11d",
            "inline-v2" to "55d910c30cd7144187e5f358dca6700ae6e7e1051eb8f7758d454f68e9144c11",
            "stable-v2" to "db29c725c70608920e600787fcda625e98429cfbae2c2d09f1dc73726825319a",
        )
    }
}
