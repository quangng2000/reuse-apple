// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

/** Internal closed set; trusted factories still pin complete graph and library contents. */
internal enum class NativeStreamingVariant(val metadata: Map<String, String>) {
    FREQUENCY_V1(mapOf(
        "execution_variant" to "native-frequency-scan-v1",
        "baseline_model_sha256" to ExperimentalNativeStreaming.BASELINE_MODEL_SHA256,
        "native_kernel_abi" to "1", "native_frequency_scans" to "16", "standard_frequency_scans" to "0",
    )),
    FREQUENCY_TEMPORAL_V2(mapOf(
        "execution_variant" to "native-frequency-temporal-v2",
        "baseline_model_sha256" to ExperimentalNativeStreaming.BASELINE_MODEL_SHA256,
        "native_kernel_abi" to "2", "native_frequency_scans" to "16", "standard_frequency_scans" to "0",
        "native_temporal_steps" to "8", "standard_temporal_steps" to "0",
    )),
    INLINE_FREQUENCY_TEMPORAL_V2(mapOf(
        "execution_variant" to "native-inline-frequency-temporal-v2",
        "baseline_model_sha256" to ExperimentalNativeStreaming.BASELINE_MODEL_SHA256,
        "native_kernel_abi" to "2", "native_frequency_scans" to "16", "standard_frequency_scans" to "0",
        "native_temporal_steps" to "8", "standard_temporal_steps" to "0",
        "inline_exp" to "ort-1.23.2-mlas-neon-fp32",
    )),
    STABLE_FREQUENCY_TEMPORAL_V2(mapOf(
        "execution_variant" to "native-stable-frequency-temporal-v2",
        "baseline_model_sha256" to ExperimentalNativeStreaming.BASELINE_MODEL_SHA256,
        "native_kernel_abi" to "2", "native_frequency_scans" to "16", "standard_frequency_scans" to "0",
        "native_temporal_steps" to "8", "standard_temporal_steps" to "0",
        "inline_exp" to "ort-1.23.2-mlas-neon-fp32", "native_softplus" to "24", "standard_softplus" to "0",
        "softplus_implementation" to "ort-1.23.2-eigen-stable-packet-log1p",
    )),
}
