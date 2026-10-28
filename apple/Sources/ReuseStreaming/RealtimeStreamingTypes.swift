/// Finite 16 kHz mono audio tagged with its original source-clock sample index.
public struct RealtimeAudioBlock: Sendable, Equatable {
    public let samples: [Float]
    public let startSample: UInt64
    public let generation: UInt64

    public init(samples: [Float], startSample: UInt64, generation: UInt64) {
        self.samples = samples
        self.startSample = startSample
        self.generation = generation
    }
}

/// Next source-clock sample to render, before applying the configured playout delay.
public struct RealtimePlaybackPosition: Sendable, Equatable {
    public let nextSample: UInt64
    public let generation: UInt64

    public init(nextSample: UInt64, generation: UInt64) {
        self.nextSample = nextSample
        self.generation = generation
    }
}

public struct RealtimeStreamingConfiguration: Sendable, Equatable {
    public let playoutDelaySamples: Int
    public let maximumPendingSamples: Int

    public init(playoutDelaySamples: Int, maximumPendingSamples: Int = 3_200) throws {
        guard playoutDelaySamples >= 0, maximumPendingSamples >= 320 else {
            throw RealtimeStreamingError.invalidConfiguration
        }
        self.playoutDelaySamples = playoutDelaySamples
        self.maximumPendingSamples = maximumPendingSamples
    }
}

public enum RealtimeStreamingError: Error, Equatable {
    case invalidConfiguration
    case nonfiniteInput
    case sampleRangeOverflow
    case overlappingInput
    case nonfiniteOutput
    case unexpectedOutputCount(expected: Int, actual: Int)
}

/// Lifetime counters. Input discarded during recovery includes the un-emitted model hop.
public struct RealtimeStreamingStatistics: Sendable, Equatable {
    public internal(set) var discardedInputSamples: UInt64 = 0
    public internal(set) var discardedOutputSamples: UInt64 = 0
    public internal(set) var deadlineResets: UInt64 = 0
    public internal(set) var overflowResets: UInt64 = 0
    public internal(set) var gapResets: UInt64 = 0
    public internal(set) var generationResets: UInt64 = 0
    public internal(set) var inferenceFailures: UInt64 = 0
    public internal(set) var processedHops: UInt64 = 0
    public internal(set) var modelResets: UInt64 = 0
}
