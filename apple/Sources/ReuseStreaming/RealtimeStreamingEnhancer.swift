/// Optional bounded live-stream coordinator. Use exclusively from one serial worker,
/// never a render callback. Not thread-safe or reentrant. All positions share a 16 kHz clock.
public final class RealtimeStreamingEnhancer {
    public var statistics: RealtimeStreamingStatistics { counters }
    public var pendingSampleCount: Int { pending.count }

    let configuration: RealtimeStreamingConfiguration
    let pushSamples: ([Float]) throws -> [Float]
    let resetModel: () -> Void
    var counters = RealtimeStreamingStatistics()
    var pending = [Float]()
    var pendingStart: UInt64?
    var expectedInputEnd: UInt64?
    var generation: UInt64?
    var latentStart: UInt64?
    var needsReset = true
    var awaitingFirstOutput = false
    var latestPlaybackSample: UInt64 = 0

    // Injection point for deterministic coordinator tests without MLX or model weights.
    init(configuration: RealtimeStreamingConfiguration,
         push: @escaping ([Float]) throws -> [Float], reset: @escaping () -> Void) {
        self.configuration = configuration
        pushSamples = push
        resetModel = reset
    }

    /// Accepts any chunk size. Invalid or overlapping input leaves state unchanged.
    /// Stale generations are discarded; a new generation invalidates the old model stream.
    public func submit(_ block: RealtimeAudioBlock) throws {
        let (end, overflow) = block.startSample.addingReportingOverflow(UInt64(block.samples.count))
        guard !overflow else { throw RealtimeStreamingError.sampleRangeOverflow }
        guard block.samples.allSatisfy({ $0.isFinite }) else { throw RealtimeStreamingError.nonfiniteInput }
        if let generation, block.generation < generation {
            increase(\.discardedInputSamples, by: UInt64(block.samples.count))
            return
        }
        if block.generation == generation, let expectedInputEnd, block.startSample < expectedInputEnd {
            throw RealtimeStreamingError.overlappingInput
        }
        advanceGeneration(block.generation)
        guard !block.samples.isEmpty else { return }
        if let expectedInputEnd, block.startSample > expectedInputEnd {
            discardStream()
            increase(\.gapResets)
        }
        if block.samples.count > configuration.maximumPendingSamples - pending.count {
            retainNewestSubmittedHop(block, end: end)
            increase(\.overflowResets)
        } else {
            if pending.isEmpty { pendingStart = block.startSample }
            pending.append(contentsOf: block.samples)
        }
        expectedInputEnd = end
    }

    /// Explicit lifecycle reset. Generations never move backwards; a same-generation
    /// reset preserves accepted-input ordering. Model reset runs before the next push.
    public func reset(generation: UInt64) {
        if let current = self.generation, generation < current { return }
        if self.generation == generation {
            discardStream()
        } else {
            advanceGeneration(generation)
        }
    }

    func advanceGeneration(_ next: UInt64) {
        guard generation != next else { return }
        if generation != nil { increase(\.generationResets) }
        discardStream()
        generation = next
        expectedInputEnd = nil
        latestPlaybackSample = 0
    }

    func observePlayback(_ position: RealtimePlaybackPosition) -> Bool {
        if let generation, position.generation < generation { return false }
        advanceGeneration(position.generation)
        latestPlaybackSample = max(latestPlaybackSample, position.nextSample)
        return true
    }

    var expiredSourcePrefix: UInt64 {
        let delay = UInt64(configuration.playoutDelaySamples)
        return latestPlaybackSample > delay ? latestPlaybackSample - delay : 0
    }

    func fullyExpired(start: UInt64) -> Bool {
        let prefix = expiredSourcePrefix
        return prefix >= start && prefix - start >= 320
    }

    func discardStream() {
        increase(\.discardedInputSamples, by: UInt64(pending.count))
        if latentStart != nil { increase(\.discardedInputSamples, by: 320) }
        pending.removeAll(keepingCapacity: false)
        pendingStart = nil
        latentStart = nil
        needsReset = true
        awaitingFirstOutput = false
    }

    func retainNewestSubmittedHop(_ block: RealtimeAudioBlock, end: UInt64) {
        // Never concatenate a potentially huge caller block with the existing queue.
        let fromPending = max(0, 320 - block.samples.count)
        var newest = Array(pending.suffix(fromPending))
        newest.append(contentsOf: block.samples.suffix(320))
        increase(\.discardedInputSamples, by: UInt64(pending.count - fromPending))
        increase(\.discardedInputSamples, by: UInt64(max(0, block.samples.count - 320)))
        if latentStart != nil { increase(\.discardedInputSamples, by: 320) }
        pending = newest
        pendingStart = end - 320
        latentStart = nil
        needsReset = true
        awaitingFirstOutput = true
    }

    func recoverDeadline() {
        let start = pendingStart! + UInt64(pending.count - 320)
        let newest = fullyExpired(start: start) ? [] : Array(pending.suffix(320))
        increase(\.discardedInputSamples, by: UInt64(pending.count - newest.count))
        if latentStart != nil { increase(\.discardedInputSamples, by: 320) }
        pending = newest
        pendingStart = newest.isEmpty ? nil : start
        latentStart = nil
        needsReset = true
        awaitingFirstOutput = !newest.isEmpty
        increase(\.deadlineResets)
    }

    func increase(_ field: WritableKeyPath<RealtimeStreamingStatistics, UInt64>, by amount: UInt64 = 1) {
        let (value, overflow) = counters[keyPath: field].addingReportingOverflow(amount)
        counters[keyPath: field] = overflow ? .max : value
    }
}
