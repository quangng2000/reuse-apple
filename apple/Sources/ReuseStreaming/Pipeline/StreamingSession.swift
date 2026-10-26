// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// See LICENSE in this module. Noncommercial research and education only.

/// Tracks framing and lifecycle independently of the network implementation.
struct StreamingSession {
    private enum Status { case open, finished, failed }
    private var status = Status.open
    private var source = StreamingFrames()
    private var receivedSamples = 0
    private var emittedSamples = 0
    private var renderedFrames = 0
    var isFresh: Bool { status == .open && receivedSamples == 0 }

    mutating func push(_ samples: [Float], render: ([Float]) throws -> [Float]) throws -> [Float] {
        try requireOpen()
        guard samples.allSatisfy({ $0.isFinite }) else { throw StreamingError.nonfiniteInput }
        var output = [Float]()
        output.reserveCapacity(samples.count)
        // Bound temporary frame construction and retained input capacity even
        // when callers supply a whole recording in one push.
        for start in stride(from: 0, to: samples.count, by: 320) {
            let chunk = Array(samples[start..<min(start + 320, samples.count)])
            receivedSamples += chunk.count
            let frames = source.push(chunk)
            output += try renderFrames(frames, render: render, limit: Int.max)
        }
        return output
    }

    mutating func finish(render: ([Float]) throws -> [Float]) throws -> [Float] {
        try requireOpen()
        guard receivedSamples >= 640 else { throw StreamingError.tooShort }
        let remaining = receivedSamples - emittedSamples
        let frames = source.finish()
        let output = try renderFrames(frames, render: render, limit: remaining)
        guard output.count == remaining else { status = .failed; throw StreamingError.invalidOutput }
        status = .finished
        return output
    }

    private mutating func renderFrames(_ frames: [[Float]], render: ([Float]) throws -> [Float],
                                       limit: Int) throws -> [Float] {
        do {
            let output = try frames.flatMap { try renderOne($0, render: render) }
            let retained = Array(output.prefix(limit))
            emittedSamples += retained.count
            return retained
        } catch {
            status = .failed
            throw error
        }
    }

    private mutating func renderOne(_ frame: [Float], render: ([Float]) throws -> [Float]) throws -> [Float] {
        let output = try render(frame)
        let expectedCount = renderedFrames == 0 ? 0 : 320
        guard output.count == expectedCount, output.allSatisfy({ $0.isFinite }) else {
            throw StreamingError.invalidOutput
        }
        renderedFrames += 1
        return output
    }

    private func requireOpen() throws {
        switch status {
        case .open: return
        case .finished: throw StreamingError.finished
        case .failed: throw StreamingError.resetRequired
        }
    }
}
