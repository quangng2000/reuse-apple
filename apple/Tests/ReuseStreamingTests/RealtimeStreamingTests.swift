import XCTest
@testable import ReuseStreaming

final class RealtimeTestEngine {
    var previous: [Float]?
    var pushCount = 0
    var pushesSinceReset = 0
    var resetCount = 0
    var onPush: (() -> Void)?
    var onReset: (() -> Void)?
    var outputOverride: [Float]?
    var failNextPush = false

    func push(_ samples: [Float]) throws -> [Float] {
        pushCount += 1
        pushesSinceReset += 1
        onPush?()
        if failNextPush {
            failNextPush = false
            throw TestFailure.inference
        }
        let output = previous.map { $0.map { $0 * 0.5 } } ?? []
        previous = samples
        return outputOverride ?? output
    }

    func reset() {
        resetCount += 1
        pushesSinceReset = 0
        previous = nil
        onReset?()
    }

    enum TestFailure: Error { case inference }
}

final class RealtimeStreamingTests: XCTestCase {
    func makeStream(_ engine: RealtimeTestEngine, delay: Int = 1_920, capacity: Int = 3_200) throws -> RealtimeStreamingEnhancer {
        RealtimeStreamingEnhancer(
            configuration: try RealtimeStreamingConfiguration(playoutDelaySamples: delay, maximumPendingSamples: capacity),
            push: engine.push, reset: engine.reset)
    }

    func block(_ start: UInt64, count: Int = 320, generation: UInt64 = 1) -> RealtimeAudioBlock {
        RealtimeAudioBlock(samples: (0..<count).map { Float((start + UInt64($0)) % 5_000 + 1) / 10_000 },
                           startSample: start, generation: generation)
    }

    func position(_ sample: UInt64 = 0, generation: UInt64 = 1) -> RealtimePlaybackPosition {
        RealtimePlaybackPosition(nextSample: sample, generation: generation)
    }

    func testHealthyChunksPreserveSourceTagsAndModelLatency() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        for index in 0..<12 {
            let start = UInt64(index * 320)
            try stream.submit(block(start))
            let output = try stream.processNext { self.position(start + 320) }
            if index == 0 {
                XCTAssertNil(output)
            } else {
                let output = try XCTUnwrap(output)
                XCTAssertEqual(output.generation, 1)
                XCTAssertEqual(output.startSample, start - 320)
                XCTAssertEqual(output.samples, block(start - 320).samples.map { $0 * 0.5 })
            }
            XCTAssertEqual(stream.pendingSampleCount, 0)
        }
        XCTAssertEqual(engine.resetCount, 1)
        XCTAssertEqual(engine.pushCount, 12)
    }

    func testPartialSubmissionsWaitForWholeHopAndEachProcessDoesOnePush() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        try stream.submit(block(0, count: 160))
        XCTAssertNil(try stream.processNext { self.position() })
        XCTAssertEqual(engine.pushCount, 0)
        try stream.submit(block(160, count: 800))
        XCTAssertNil(try stream.processNext { self.position() })
        XCTAssertEqual(engine.pushCount, 1)
        XCTAssertEqual(stream.pendingSampleCount, 640)
        let output = try XCTUnwrap(stream.processNext { self.position() })
        XCTAssertEqual(engine.pushCount, 2)
        XCTAssertEqual(output.startSample, 0)
        XCTAssertEqual(output.samples, block(0).samples.map { $0 * 0.5 })
    }

    func testOverflowRetainsNewestHopWithoutRelabelingOldAudio() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine, capacity: 640)
        try stream.submit(block(0))
        _ = try stream.processNext { self.position() }
        try stream.submit(block(320, count: 10_000))
        XCTAssertEqual(stream.pendingSampleCount, 320)
        XCTAssertNil(try stream.processNext { self.position(10_320) })
        try stream.submit(block(10_320))
        let output = try XCTUnwrap(stream.processNext { self.position(10_640) })
        XCTAssertEqual(output.startSample, 10_000)
        XCTAssertEqual(output.samples, block(10_000).samples.map { $0 * 0.5 })
        XCTAssertEqual(engine.resetCount, 2)
        XCTAssertEqual(stream.statistics.overflowResets, 1)
    }

    func testExpiredNewestHopSkipsInferenceUntilFreshInputArrives() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        try stream.submit(block(0, count: 960))
        XCTAssertNil(try stream.processNext { self.position(12_000) })
        XCTAssertEqual(engine.pushCount, 0)
        XCTAssertEqual(stream.pendingSampleCount, 0)
        XCTAssertNil(try stream.processNext { self.position(12_000) })
        XCTAssertEqual(engine.pushCount, 0)
        try stream.submit(block(12_000))
        XCTAssertNil(try stream.processNext { self.position(12_320) })
        try stream.submit(block(12_320))
        let output = try XCTUnwrap(stream.processNext { self.position(12_640) })
        XCTAssertEqual(output.startSample, 12_000)
        XCTAssertEqual(output.samples, block(12_000).samples.map { $0 * 0.5 })
    }

    func testExpiredLatentHopJumpsToFreshInputBeforeQueueCapacityIsReached() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        try stream.submit(block(0))
        _ = try stream.processNext { self.position() }
        try stream.submit(block(320))
        XCTAssertNotNil(try stream.processNext { self.position() })
        // Complete startup before testing expiry, so the cold-start latch is released.
        // The pending input is below capacity, but only its newest hop is still fresh.
        try stream.submit(block(640, count: 2_560))
        XCTAssertNil(try stream.processNext { self.position(4_800) })
        XCTAssertEqual(engine.previous, block(2_880).samples)
        XCTAssertEqual(stream.pendingSampleCount, 0)
        try stream.submit(block(3_200))
        let output = try XCTUnwrap(stream.processNext { self.position(4_800) })
        XCTAssertEqual(output.startSample, 2_880)
        XCTAssertEqual(output.samples, block(2_880).samples.map { $0 * 0.5 })
    }

    func testForwardGapDiscardsLatentAudioAndResetsBeforeFreshOutput() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        try stream.submit(block(0))
        _ = try stream.processNext { self.position() }
        try stream.submit(block(1_000))
        XCTAssertNil(try stream.processNext { self.position(1_320) })
        try stream.submit(block(1_320))
        let output = try XCTUnwrap(stream.processNext { self.position(1_640) })
        XCTAssertEqual(output.startSample, 1_000)
        XCTAssertEqual(output.samples, block(1_000).samples.map { $0 * 0.5 })
        XCTAssertEqual(engine.resetCount, 2)
        XCTAssertEqual(stream.statistics.gapResets, 1)
    }

    func testSilenceDoesNotRequestModelRecovery() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        for index in 0..<200 {
            let start = UInt64(index * 320)
            try stream.submit(RealtimeAudioBlock(samples: Array(repeating: 0, count: 320), startSample: start, generation: 1))
            let output = try stream.processNext { self.position(start + 320) }
            if index > 0 { XCTAssertEqual(output?.samples, Array(repeating: 0, count: 320)) }
        }
        XCTAssertEqual(engine.resetCount, 1)
        XCTAssertEqual(stream.statistics.deadlineResets, 0)
    }
}
