// Apple audio import/export wrapper. SPDX-License-Identifier: OpenMDW-1.1
import AVFoundation
import Foundation

/// Apple audio-file conversion for mono/stereo recordings up to 60 seconds.
public enum AudioFileIO {
    /// Converts supported audio to 16 kHz mono Float32 samples.
    public static func read(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        try validate(file)
        let format = file.processingFormat
        let source = try readFrames(file)
        if format.sampleRate == 16000 && format.channelCount == 1,
           let pointer = source.floatChannelData?[0] {
            return Array(UnsafeBufferPointer(start: pointer, count: Int(source.frameLength)))
        }
        return try AudioConversion.convert(source)
    }
    private static func validate(_ file: AVAudioFile) throws {
        let format = file.processingFormat
        let seconds = Double(file.length) / format.sampleRate
        guard seconds > 0, seconds <= 60, format.channelCount <= 2,
              format.sampleRate >= 8000, format.sampleRate <= 192000 else {
            throw ReuseError.invalid("Choose a mono or stereo recording of up to 60 seconds (8–192 kHz).")
        }
    }
    private static func readFrames(_ file: AVAudioFile) throws -> AVAudioPCMBuffer {
        guard let source = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
              let chunk = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 4096) else {
            throw ReuseError.invalid("Cannot allocate an audio read buffer.")
        }
        // AVAudioFile can return partial reads; keep reading the declared length.
        var readFrames = 0
        while readFrames < Int(file.length) {
            let wanted = min(4096, Int(file.length)-readFrames)
            try file.read(into: chunk, frameCount: AVAudioFrameCount(wanted))
            let count = Int(chunk.frameLength)
            guard count > 0 else { throw ReuseError.invalid("Audio ended before all frames were read.") }
            copyChannels(chunk, to: source, offset: readFrames)
            readFrames += count
        }
        source.frameLength = AVAudioFrameCount(readFrames)
        return source
    }
    private static func copyChannels(_ chunk: AVAudioPCMBuffer, to source: AVAudioPCMBuffer, offset: Int) {
        for channel in 0..<Int(source.format.channelCount) {
            source.floatChannelData![channel].advanced(by: offset)
                .update(from: chunk.floatChannelData![channel], count: Int(chunk.frameLength))
        }
    }
    /// Writes finite 16 kHz mono Float32 WAV audio at the requested URL.
    public static func write(_ samples: [Float], to url: URL) throws {
        guard samples.allSatisfy({ $0.isFinite }), !samples.isEmpty,
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw ReuseError.invalid("Cannot save invalid audio.")
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in buffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count) }
        let file = try AVAudioFile(forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        try file.write(from: buffer)
    }
}
