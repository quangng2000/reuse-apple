import AVFoundation
import ReuseAudio

enum MicrophoneConversion {
    static func convert(_ source: AVAudioPCMBuffer, using converter: AVAudioConverter) throws -> [Float] {
        let capacity = AVAudioFrameCount(ceil(Double(source.frameLength) * 16000 / source.format.sampleRate) + 64)
        guard let output = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: capacity) else {
            throw ReuseError.invalid("Cannot allocate a microphone conversion buffer.")
        }
        var consumed = false, error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            status.pointee = consumed ? .noDataNow : .haveData
            defer { consumed = true }
            return consumed ? nil : source
        }
        if let error { throw error }
        guard let pointer = output.floatChannelData?[0], output.frameLength > 0 else { return [] }
        return Array(UnsafeBufferPointer(start: pointer, count: Int(output.frameLength)))
    }

    static func finish(using converter: AVAudioConverter, onSamples: ([Float]) -> Void) throws {
        guard let output = AVAudioPCMBuffer(pcmFormat: converter.outputFormat, frameCapacity: 4096) else {
            throw ReuseError.invalid("Cannot allocate the final microphone conversion buffer.")
        }
        while true {
            output.frameLength = 0
            var error: NSError?
            let status = converter.convert(to: output, error: &error) { _, state in
                state.pointee = .endOfStream; return nil
            }
            if let error { throw error }
            if let pointer = output.floatChannelData?[0], output.frameLength > 0 {
                onSamples(Array(UnsafeBufferPointer(start: pointer, count: Int(output.frameLength))))
            }
            if status == .endOfStream { return }
            if status == .error || output.frameLength == 0 { throw ReuseError.invalid("Microphone conversion could not finish.") }
        }
    }
}
