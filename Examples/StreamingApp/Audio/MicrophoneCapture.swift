import AVFoundation
import ReuseAudio

final class MicrophoneCapture {
    private let engine = AVAudioEngine()
    private let lock = NSLock()
    private var acceptsAudio = false
    private var tapInstalled = false
    private var converter: AVAudioConverter?
    private var deliver: (([Float]) -> Void)?

    func start(onSamples: @escaping ([Float]) -> Void, onFailure: @escaping (String) -> Void) throws {
        let converter = try configure()
        self.converter = converter
        deliver = onSamples
        converter.primeMethod = .none
        lock.lock(); acceptsAudio = true; lock.unlock()
        let frames = AVAudioFrameCount(converter.inputFormat.sampleRate * 0.02)
        engine.inputNode.installTap(onBus: 0, bufferSize: frames, format: nil) { buffer, _ in
            self.lock.lock(); defer { self.lock.unlock() }
            guard self.acceptsAudio else { return }
            do { onSamples(try MicrophoneConversion.convert(buffer, using: converter)) }
            catch { onFailure(error.localizedDescription) }
        }
        tapInstalled = true
        do { engine.prepare(); try engine.start() }
        catch { stop(); throw error }
    }

    private func configure() throws -> AVAudioConverter {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
        try session.setPreferredSampleRate(16000)
        try session.setPreferredIOBufferDuration(0.02)
        try session.setActive(true)
        let input = engine.inputNode.outputFormat(forBus: 0)
        guard let output = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                        channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: input, to: output) else {
            throw ReuseError.invalid("Cannot convert microphone audio to 16 kHz mono.")
        }
        return converter
    }

    /// Normal Stop delivers buffered resampler audio before the model is flushed.
    func finishCapture() throws {
        stopEngine()
        defer { converter = nil; deliver = nil }
        if let converter, let deliver { try MicrophoneConversion.finish(using: converter, onSamples: deliver) }
    }

    /// Cancellation deliberately discards pending converter audio.
    func stop() {
        stopEngine()
        converter = nil; deliver = nil
    }

    private func stopEngine() {
        lock.lock(); acceptsAudio = false; lock.unlock()
        engine.stop()
        if tapInstalled { engine.inputNode.removeTap(onBus: 0); tapInstalled = false }
    }
}
