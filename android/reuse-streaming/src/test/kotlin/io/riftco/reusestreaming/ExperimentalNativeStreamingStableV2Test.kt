package io.riftco.reusestreaming

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.file.Files

class ExperimentalNativeStreamingStableV2Test {
    private fun json(name: String) = JSONObject(javaClass.getResource("/$name")!!.readText())
    private fun temporary(block: (File) -> Unit) {
        val folder = Files.createTempDirectory("experimental-native-streaming").toFile()
        try { block(folder) } finally { folder.deleteRecursively() }
    }

    @Test fun nonEightLayerRequestIsRejectedBeforeAnyFileOrLibraryAccess() {
        assertThrows(IllegalArgumentException::class.java) {
            ExperimentalNativeStreamingStableV2.create(File("absent.onnx"), File("absent.json"),
                File("absent.so"), File("absent.so.json"), StreamingEnhancerConfiguration(4))
        }
    }

    @Test fun everyPinnedNativeGraphIdentityFieldIsValidatedBeforeModelOrLibraryLoading() = temporary { folder ->
        val edits: List<(JSONObject) -> Unit> = listOf(
            { it.put("modelSha256", "0".repeat(64)) }, { it.put("modelBytes", 1) },
            { it.put("executionVariant", "unknown") }, { it.put("baselineModelSha256", "0".repeat(64)) },
            { it.put("customOpDomain", "unknown") }, { it.put("customOpName", "unknown") },
            { it.put("customTemporalOpName", "unknown") },
            { it.put("inlineExp", "different-exp") },
            { it.put("nativeInlineBaselineModelSha256", "0".repeat(64)) },
            { it.put("customSoftplusOpName", "unknown") },
            { it.put("nativeSoftplusCount", 23) }, { it.put("standardSoftplusCount", 1) },
            { it.put("softplusImplementation", "log-one-plus-exp") },
            { it.put("nativeTemporalBaselineModelSha256", "0".repeat(64)) },
            { it.getJSONArray("nativeTargets").put(0, "x86_64") },
            { it.put("nativeFrequencyBaselineModelSha256", "0".repeat(64)) },
            { it.put("kernelAbi", 3) }, { it.put("kernelAbi", "1") },
            { it.put("nativeFrequencyScanCount", 15) }, { it.put("standardFrequencyScanCount", 1) },
            { it.put("preparedDecay", true) }, { it.put("preparedGate", false) },
            { it.put("nativeTemporalStepCount", 7) }, { it.put("standardTemporalStepCount", 1) },
            { it.put("temporalPreparedDecay", true) }, { it.put("temporalPreparedGate", false) },
            { it.put("temporalReduction", "parallel") },
            { it.put("recurrenceReduction", "parallel") }, { it.put("nativeChannelParallel", false) },
            { it.getJSONObject("nativeSources").put("native_stable_v2.cpp", "0".repeat(64)) },
            { it.getJSONObject("nativeSources").put("extra.cpp", "0".repeat(64)) },
        )
        for (edit in edits) {
            val receipt = json("streaming-native-stable-v2-manifest-l8.json").also(edit)
            val manifest = File(folder, "model.json").also { it.writeText(receipt.toString()) }
            val failure = assertThrows(IllegalArgumentException::class.java) {
                ExperimentalNativeStreamingStableV2.create(File("absent.onnx"), manifest, File("absent.so"),
                    File("absent.so.json"), StreamingEnhancerConfiguration(8))
            }
            assertFalse("Must reject the contract before reaching the absent model", failure.message!!.contains("size check"))
        }
    }

    @Test fun nativeLibraryReceiptRejectsIdentityTypeAbiAndSourceChangesBeforeLoading() = temporary { folder ->
        val edits: List<(JSONObject) -> Unit> = listOf(
            { it.put("kernelAbi", 3) }, { it.put("kernelAbi", 1.5) }, { it.put("domain", "other") },
            { it.put("op", "other") }, { it.put("temporalOp", "other") }, { it.put("softplusOp", "other") }, { it.put("ortVersion", "1.24.0") }, { it.put("abi", "x86_64") },
            { it.put("precision", "float16") }, { it.put("sourceSha256", "0".repeat(64)) },
            { it.put("compilerFlags", "-ffast-math") }, { it.put("file", "other.so") },
            { it.put("bytes", 1) }, { it.put("sha256", "0".repeat(64)) },
            { it.getJSONObject("sources").put("native_stable_v2.cpp", "0".repeat(64)) },
            { it.getJSONObject("sources").remove("transform_stable_v2.py") },
            { it.getJSONArray("verifiedExportedSymbols").put(0, "WrongRegistrationSymbol") },
            { it.getJSONArray("verifiedExportedSymbols").remove(1) },
        )
        for (edit in edits) {
            val receipt = json("streaming-native-stable-v2-library-arm64.json").also(edit)
            val file = File(folder, "library.json").also { it.writeText(receipt.toString()) }
            assertThrows(IllegalArgumentException::class.java) {
                ExperimentalNativeStreamingStableV2.verifyLibrary(File(folder, "libreuse_streaming_stable_v2.so"), file)
            }
        }
    }

    @Test fun correctlySizedButCorruptedLibraryFailsShaBeforeAnyRegistration() = temporary { folder ->
        val receipt = json("streaming-native-stable-v2-library-arm64.json")
        val file = File(folder, "library.json").also { it.writeText(receipt.toString()) }
        val binary = File(folder, "libreuse_streaming_stable_v2.so").also { it.writeBytes(ByteArray(receipt.getInt("bytes"))) }
        val failure = assertThrows(IllegalArgumentException::class.java) {
            ExperimentalNativeStreamingStableV2.verifyLibrary(binary, file)
        }
        assertEquals("Experimental native library integrity check failed.", failure.message)
        binary.appendBytes(byteArrayOf(0))
        assertThrows(IllegalArgumentException::class.java) { ExperimentalNativeStreamingStableV2.verifyLibrary(binary, file) }
    }

    @Test fun missingOversizedAndMalformedReceiptsAreRejectedWithoutLoadingCode() = temporary { folder ->
        val file = File(folder, "bad.json")
        for (text in listOf("", " ".repeat(262_145), "{", "[]")) {
            file.writeText(text)
            assertThrows(Exception::class.java) {
                ExperimentalNativeStreamingStableV2.verifyLibrary(File(folder, "libreuse_streaming_stable_v2.so"), file)
            }
        }
    }
}
