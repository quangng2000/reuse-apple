// Apple audio import wrapper. SPDX-License-Identifier: OpenMDW-1.1
import AVFoundation

private final class ConversionInputState {
    var sent = false
}

enum AudioConversion {
    static func convert(_ source: AVAudioPCMBuffer) throws -> [Float] {
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: source.format, to: format),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4096) else {
            throw ReuseError.invalid("This audio format cannot be converted.")
        }
        let state = ConversionInputState()
        var samples = [Float]()
        while true {
            let status = try nextBuffer(source, converter: converter, buffer: buffer, state: state)
            if buffer.frameLength > 0, let pointer = buffer.floatChannelData?[0] {
                samples += Array(UnsafeBufferPointer(start: pointer, count: Int(buffer.frameLength)))
            }
            if status == .endOfStream { break }
            try validate(status, frames: buffer.frameLength)
        }
        return samples
    }
    private static func nextBuffer(_ source: AVAudioPCMBuffer, converter: AVAudioConverter,
                                   buffer: AVAudioPCMBuffer, state input: ConversionInputState) throws -> AVAudioConverterOutputStatus {
        var error: NSError?
        buffer.frameLength = 0
        let status = converter.convert(to: buffer, error: &error) { _, state in
            if input.sent { state.pointee = .endOfStream; return nil }
            input.sent = true
            state.pointee = .haveData
            return source
        }
        if let error { throw error }
        return status
    }
    private static func validate(_ status: AVAudioConverterOutputStatus, frames: AVAudioFrameCount) throws {
        if status == .error || (status == .inputRanDry && frames == 0) {
            throw ReuseError.invalid("Audio conversion ended unexpectedly.")
        }
    }
}
