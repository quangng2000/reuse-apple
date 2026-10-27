extension RealtimeStreamingEnhancer {
    /// Processes at most one 320-sample hop. Refresh host ingress between calls.
    /// The host must still revalidate returned tags before rendering: the clock keeps moving.
    public func processNext(playbackPosition: () -> RealtimePlaybackPosition) throws -> RealtimeAudioBlock? {
        guard observePlayback(playbackPosition()), pending.count >= 320 else { return nil }
        let nextOutput = latentStart ?? pendingStart!
        if (!awaitingFirstOutput || needsReset) && fullyExpired(start: nextOutput) { recoverDeadline() }
        guard pending.count >= 320 else { return nil }
        if needsReset {
            let resetGeneration = generation!
            resetModel()
            increase(\.modelResets)
            needsReset = false
            awaitingFirstOutput = false
            // Reset can run synchronous GPU work too. Its entry clock cannot
            // authorize a later push after the generation or deadline changes.
            guard observePlayback(playbackPosition()), generation == resetGeneration else { return nil }
            if fullyExpired(start: latentStart ?? pendingStart!) {
                recoverDeadline()
                return nil
            }
            awaitingFirstOutput = true
        }
        let pushedStart = pendingStart!
        let pushedGeneration = generation!
        let outputStart = latentStart
        let input = Array(pending.prefix(320))
        pending.removeFirst(320)
        pendingStart = pending.isEmpty ? nil : pushedStart + 320
        let output: [Float]
        do {
            output = try pushSamples(input)
        } catch {
            inferenceFailed(playback: playbackPosition())
            throw error
        }
        let after = playbackPosition()
        let expected = outputStart == nil ? 0 : 320
        do {
            guard output.allSatisfy({ $0.isFinite }) else { throw RealtimeStreamingError.nonfiniteOutput }
            guard output.count == expected else {
                throw RealtimeStreamingError.unexpectedOutputCount(expected: expected, actual: output.count)
            }
        } catch {
            increase(\.discardedOutputSamples, by: UInt64(output.count))
            inferenceFailed(playback: after)
            throw error
        }
        latentStart = pushedStart
        increase(\.processedHops)
        // A completed output releases the latch even if late; otherwise a slow
        // stream could remain latched forever and never jump to fresh input.
        if !output.isEmpty { awaitingFirstOutput = false }
        guard observePlayback(after), generation == pushedGeneration else {
            increase(\.discardedOutputSamples, by: UInt64(output.count))
            return nil
        }
        guard let outputStart else { return nil }
        let prefix = expiredSourcePrefix
        let removed = prefix > outputStart ? Int(min(UInt64(output.count), prefix - outputStart)) : 0
        increase(\.discardedOutputSamples, by: UInt64(removed))
        guard removed < output.count else { return nil }
        return RealtimeAudioBlock(samples: Array(output.dropFirst(removed)),
                                  startSample: outputStart + UInt64(removed), generation: pushedGeneration)
    }

    private func inferenceFailed(playback: RealtimePlaybackPosition) {
        // The current push may have consumed input before throwing. Discard it,
        // the older latent hop, and pending input before resetting the vendor model.
        increase(\.inferenceFailures)
        increase(\.discardedInputSamples, by: 320)
        discardStream()
        _ = observePlayback(playback)
    }
}
