// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

/** Lifecycle errors. TOO_SHORT and NONFINITE_INPUT leave the stream open and unchanged. */
class StreamingException(val reason: Reason, cause: Throwable? = null) : IllegalStateException(reason.message, cause) {
    enum class Reason(internal val message: String) {
        NONFINITE_INPUT("Supply finite 16 kHz mono PCM."),
        TOO_SHORT("Supply at least 640 samples (40 ms) before finishing."),
        FINISHED("The stream is finished. Reset before sending more audio."),
        RESET_REQUIRED("Streaming inference failed or was cancelled. Reset before continuing."),
        INVALID_OUTPUT("The model returned invalid or incomplete audio. Reset before continuing."),
        WARMUP_REQUIRES_RESET("Warm up before sending audio, or reset first."),
    }
}
