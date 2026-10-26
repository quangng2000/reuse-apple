// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

/** CPU FP32 inference only. The model file must be exported for this exact layer count. */
data class StreamingEnhancerConfiguration(val layers: Int = 4, val intraOpThreads: Int = 2) {
    init {
        require(layers in 3..12) { "Choose a streaming exit between 3 and 12 layers." }
        require(intraOpThreads in 1..8) { "Choose 1–8 CPU inference threads." }
    }
}
