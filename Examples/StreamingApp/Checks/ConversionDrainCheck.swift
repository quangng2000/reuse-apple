import AVFoundation

enum ReuseError: Error { case invalid(String) }

@main
enum ConversionDrainCheck {
    static func main() throws {
        for rate in [16000.0, 44100.0, 48000.0] {
            for count in [1001, Int(rate) + 137] {
                let source = (0..<count).map { Float(sin(Double($0) * 0.13) * 0.2) }
                let whole = try render(source, rate: rate, chunkSize: source.count)
                let chunked = try render(source, rate: rate, chunkSize: 197)
                guard whole.samples.count == chunked.samples.count,
                      chunked.samples.allSatisfy({ $0.isFinite }),
                      zip(whole.samples, chunked.samples).allSatisfy({ abs($0 - $1) < 0.00001 }) else {
                    throw ReuseError.invalid("Conversion differs across partitions at \(rate) Hz.")
                }
                print("PASS rate=\(Int(rate)) input=\(count) output=\(chunked.samples.count) drained=\(chunked.tail)")
            }
        }
    }

    private static func render(_ samples: [Float], rate: Double, chunkSize: Int) throws -> (samples: [Float], tail: Int) {
        let input = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false)!
        let output = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
        let converter = AVAudioConverter(from: input, to: output)!
        converter.primeMethod = .none
        var result = [Float]()
        for start in stride(from: 0, to: samples.count, by: chunkSize) {
            let part = Array(samples[start..<min(samples.count, start + chunkSize)])
            let buffer = AVAudioPCMBuffer(pcmFormat: input, frameCapacity: AVAudioFrameCount(part.count))!
            buffer.frameLength = buffer.frameCapacity
            part.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: part.count) }
            result += try MicrophoneConversion.convert(buffer, using: converter)
        }
        let before = result.count
        try MicrophoneConversion.finish(using: converter) { result += $0 }
        return (result, result.count - before)
    }
}
