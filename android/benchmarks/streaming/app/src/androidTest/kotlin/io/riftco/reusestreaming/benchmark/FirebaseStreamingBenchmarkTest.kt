package io.riftco.reusestreaming.benchmark

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest

/** Launches the existing nondebuggable harness; inference is timed only inside MainActivity. */
@RunWith(AndroidJUnit4::class)
class FirebaseStreamingBenchmarkTest {
    @Test
    fun measureEightLayerStreaming() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val arguments = InstrumentationRegistry.getArguments()
        val context = instrumentation.targetContext
        val runId = arguments.getString("benchmarkRunId") ?: "firebase-${System.currentTimeMillis()}"
        require(runId.matches(Regex("[A-Za-z0-9_.-]{1,80}"))) { "Unsafe benchmarkRunId." }
        val frames = intArgument(arguments, "benchmarkFrames", 300, 32..1500)
        val threads = intArgument(arguments, "benchmarkThreads", 4, 1..8)
        val capture = boolArgument(arguments, "benchmarkCaptureOutput", false)
        val requirePhysical = boolArgument(arguments, "benchmarkRequirePhysical", true)
        val expectedVariant = arguments.getString("benchmarkExpectedVariant")
        val expectedSource = arguments.getString("benchmarkExpectedSdkSource")
        val expectedModel = arguments.getString("benchmarkExpectedModelSha256")
        val expectedInput = arguments.getString("benchmarkExpectedInputSha256")
        for (hash in listOfNotNull(expectedModel, expectedInput)) {
            require(hash.matches(Regex("[a-f0-9]{64}"))) { "Expected SHA256 must be lowercase hexadecimal." }
        }
        val resultFile = File(context.filesDir, "result-$runId.json")
        check(!resultFile.exists() || resultFile.delete()) { "Could not remove stale owned result." }
        val external = checkNotNull(context.getExternalFilesDir(null)) { "App external storage unavailable." }
        val artifacts = File(external, "firebase-benchmark").apply { check(mkdirs() || isDirectory) }
        val receiptFile = File(artifacts, "$runId-receipt.json")
        val receipt = JSONObject().put("schemaVersion", 1).put("runId", runId)
            .put("instrumentationClass", javaClass.name).put("success", false)
            .put("measurementScope", "In-app CPU FP32 streaming push latency; excludes cloud scheduling, install, network and remote-control latency")
            .put("requested", JSONObject().put("layers", 8).put("threads", threads).put("timedFrames", frames)
                .put("captureOutput", capture).put("requirePhysical", requirePhysical)
                .put("expectedVariant", expectedVariant ?: JSONObject.NULL)
                .put("expectedSdkSource", expectedSource ?: JSONObject.NULL)
                .put("expectedModelSha256", expectedModel ?: JSONObject.NULL)
                .put("expectedInputSha256", expectedInput ?: JSONObject.NULL))
        var activity: Activity? = null
        val started = SystemClock.elapsedRealtimeNanos()
        try {
            val launch = Intent().setClassName(context.packageName, "${context.packageName}.MainActivity")
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                .putExtra("runId", runId).putExtra("timedFrames", frames)
                .putExtra("primaryOnly", true).putExtra("layers", "8").putExtra("threads", threads.toString())
                .putExtra("captureOutput", capture)
            arguments.getString("benchmarkInputAsset")?.let { launch.putExtra("inputAsset", it) }
            activity = instrumentation.startActivitySync(launch)
            // No UI automation, tracing, shell calls or sampling are performed during inference.
            // The sleeping test thread checks for the single final file once per second.
            val deadline = SystemClock.elapsedRealtime() + 660_000L
            var parsed: JSONObject? = null
            var raw: ByteArray? = null
            while (SystemClock.elapsedRealtime() < deadline) {
                if (resultFile.isFile && resultFile.length() > 0) {
                    val candidate = resultFile.readBytes()
                    val json = try { JSONObject(candidate.toString(Charsets.UTF_8)) } catch (_: org.json.JSONException) { null }
                    if (json != null && json.optString("runId") == runId && json.optString("status") in setOf("complete", "failed")) {
                        raw = candidate
                        parsed = json
                        break
                    }
                }
                SystemClock.sleep(1_000L)
            }
            val result = checkNotNull(parsed) { "No final benchmark JSON within the bounded eleven-minute instrumentation wait." }
            val rawResult = checkNotNull(raw)
            val artifact = File(artifacts, "$runId-result.json")
            artifact.writeBytes(rawResult)
            receipt.put("result", JSONObject().put("file", artifact.absolutePath)
                .put("sha256", sha256(rawResult)).put("bytes", rawResult.size))
            check(result.getBoolean("success") && result.getString("status") == "complete") {
                "Benchmark failed: ${result.optJSONObject("error")?.optString("message") ?: "inspect raw JSON"}"
            }
            check(result.getString("precision") == "FP32" && result.getString("executionProvider") == "CPUExecutionProvider")
            check(!result.getJSONObject("device").getBoolean("targetDebuggable")) { "Target is debuggable." }
            if (requirePhysical) check(!result.getJSONObject("device").getBoolean("emulatorReported")) { "Physical-device benchmark requested; emulator reported." }
            val abis = result.getJSONObject("device").getJSONArray("abis")
            check((0 until abis.length()).any { abis.getString(it) == "arm64-v8a" }) { "ARM64 target required." }
            expectedVariant?.let { check(result.getString("variant") == it) { "Unexpected build variant." } }
            expectedSource?.let { check(result.getString("sdkSource") == it) { "Unexpected SDK source." } }
            expectedInput?.let {
                check(result.optString("inputSha256", result.optString("syntheticInputSha256")) == it) { "Input SHA256 mismatch." }
            }
            val cases = result.getJSONArray("cases")
            check(cases.length() == 1) { "Exactly one eight-layer case is required per execution." }
            val case = cases.getJSONObject(0)
            check(case.getBoolean("success") && case.getInt("layers") == 8 && case.getInt("threads") == threads && case.getInt("timedFrames") == frames)
            check(case.getBoolean("allOutputFinite") && case.getInt("inputSamples") == case.getInt("outputSamples"))
            check(case.getJSONObject("fixture").getBoolean("passed") && case.getJSONObject("fixture").getBoolean("resetBitwiseRepeatPassed"))
            expectedModel?.let { check(case.getString("modelSha256") == it) { "Model SHA256 mismatch." } }
            check(case.optBoolean("qualityCapture", false) == capture) { "Unexpected quality capture mode." }
            val times = case.getJSONArray("rawPushMs")
            check(times.length() == frames)
            for (index in 0 until times.length()) check(times.getDouble(index).let { it.isFinite() && it >= 0.0 })
            if (capture) {
                val exported = case.getJSONObject("export")
                for (kind in listOf("input", "output")) {
                    val pcm = exported.getJSONObject(kind)
                    val source = File(pcm.getString("file"))
                    val ownedFolder = File(external, "benchmark-export").canonicalFile
                    check(source.canonicalFile.parentFile == ownedFolder && source.name == "$runId-${case.getString("caseId")}-$kind.f32")
                    val bytes = source.readBytes()
                    check(bytes.size == pcm.getInt("bytes") && sha256(bytes) == pcm.getString("sha256"))
                    File(artifacts, source.name).writeBytes(bytes)
                }
            }
            receipt.put("success", true)
        } catch (failure: Throwable) {
            receipt.put("error", JSONObject().put("type", failure.javaClass.name).put("message", failure.message ?: "No message"))
            throw failure
        } finally {
            receipt.put("elapsedMs", (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0)
            receiptFile.writeText(receipt.toString(2), Charsets.UTF_8)
            Log.i("REUSE_FIREBASE", "ARTIFACT_RECEIPT ${receiptFile.absolutePath} sha256=${sha256(receiptFile.readBytes())}")
            activity?.let { launched -> instrumentation.runOnMainSync { launched.finish() } }
        }
    }

    private fun intArgument(arguments: Bundle, name: String, default: Int, allowed: IntRange): Int {
        val value = arguments.getString(name)?.let { it.toIntOrNull() ?: error("Invalid $name integer.") } ?: default
        require(value in allowed) { "$name outside $allowed." }
        return value
    }

    private fun boolArgument(arguments: Bundle, name: String, default: Boolean): Boolean =
        arguments.getString(name)?.let { value ->
            when (value) { "true" -> true; "false" -> false; else -> error("$name must be true or false.") }
        } ?: default

    private fun sha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes)
        .joinToString("") { "%02x".format(it.toInt() and 255) }
}
