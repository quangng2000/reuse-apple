import XCTest
@testable import ReuseStreaming

extension RealtimeStreamingTests {
    func testHopThatExpiresDuringModelResetDoesNotStartInference() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        var clock = position()
        engine.onReset = { clock = self.position(12_000) }
        try stream.submit(block(0))
        XCTAssertNil(try stream.processNext { clock })
        XCTAssertEqual(engine.pushCount, 0)
        XCTAssertEqual(stream.pendingSampleCount, 0)
        engine.onReset = nil
        try stream.submit(block(12_000))
        XCTAssertNil(try stream.processNext { clock })
        try stream.submit(block(12_320))
        XCTAssertEqual(try stream.processNext { clock }?.startSample, 12_000)
    }

    func testGenerationChangeDuringModelResetDoesNotStartOldInference() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        var clock = position()
        engine.onReset = { clock = self.position(generation: 2) }
        try stream.submit(block(0))
        XCTAssertNil(try stream.processNext { clock })
        XCTAssertEqual(engine.pushCount, 0)
        XCTAssertEqual(stream.pendingSampleCount, 0)
        engine.onReset = nil
        try stream.submit(block(0, generation: 2))
        XCTAssertNil(try stream.processNext { clock })
        try stream.submit(block(320, generation: 2))
        XCTAssertEqual(try stream.processNext { clock }?.generation, 2)
    }

    func testGenerationChangeDuringInferenceSuppressesOldOutput() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        var clock = position()
        try stream.submit(block(0))
        _ = try stream.processNext { clock }
        try stream.submit(block(320))
        engine.onPush = { clock = self.position(generation: 2) }
        XCTAssertNil(try stream.processNext { clock })
        engine.onPush = nil
        try stream.submit(block(640, generation: 1)) // Late old-call input cannot revive generation 1.
        XCTAssertEqual(stream.pendingSampleCount, 0)
        try stream.submit(block(0, generation: 2))
        XCTAssertNil(try stream.processNext { clock })
        try stream.submit(block(320, generation: 2))
        let output = try XCTUnwrap(stream.processNext { clock })
        XCTAssertEqual(output.generation, 2)
        XCTAssertEqual(output.startSample, 0)
        XCTAssertEqual(output.samples, block(0).samples.map { $0 * 0.5 })
    }

    func testExplicitResetDoesNotPermitGenerationRollback() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        stream.reset(generation: 3)
        stream.reset(generation: 2)
        try stream.submit(block(0, generation: 2))
        XCTAssertEqual(stream.pendingSampleCount, 0)
        try stream.submit(block(0, generation: 3))
        XCTAssertNil(try stream.processNext { self.position(generation: 2) })
        XCTAssertEqual(engine.pushCount, 0)
        XCTAssertNil(try stream.processNext { self.position(generation: 3) })
        try stream.submit(block(320, generation: 3))
        XCTAssertEqual(try stream.processNext { self.position(generation: 3) }?.generation, 3)
    }

    func testClockIsReadAfterInferenceAndExpiredPrefixIsTrimmed() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        var clock = position()
        try stream.submit(block(0))
        _ = try stream.processNext { clock }
        try stream.submit(block(320))
        engine.onPush = { clock = self.position(1_920 + 160) }
        let output = try XCTUnwrap(stream.processNext { clock })
        XCTAssertEqual(output.startSample, 160)
        XCTAssertEqual(output.samples, Array(block(0).samples.suffix(160)).map { $0 * 0.5 })
        XCTAssertEqual(stream.statistics.discardedOutputSamples, 160)
    }

    func testFullyExpiredInFlightOutputIsNotPublishedAndCanRecover() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        var clock = position()
        try stream.submit(block(0))
        _ = try stream.processNext { clock }
        try stream.submit(block(320))
        engine.onPush = { clock = self.position(12_000) }
        XCTAssertNil(try stream.processNext { clock })
        engine.onPush = nil
        try stream.submit(block(12_000))
        XCTAssertNil(try stream.processNext { clock })
        try stream.submit(block(12_320))
        let output = try XCTUnwrap(stream.processNext { clock })
        XCTAssertEqual(output.startSample, 12_000)
        XCTAssertEqual(output.samples, block(12_000).samples.map { $0 * 0.5 })
    }

    func testInvalidInputDoesNotChangePendingOrGeneration() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        try stream.submit(block(0))
        XCTAssertThrowsError(try stream.submit(RealtimeAudioBlock(samples: [.nan], startSample: 0, generation: 2)))
        XCTAssertThrowsError(try stream.submit(RealtimeAudioBlock(samples: [1], startSample: UInt64.max, generation: 2)))
        XCTAssertThrowsError(try stream.submit(block(160)))
        XCTAssertEqual(stream.pendingSampleCount, 320)
        XCTAssertNil(try stream.processNext { self.position() })
        try stream.submit(block(320))
        XCTAssertEqual(try stream.processNext { self.position() }?.startSample, 0)
    }

    func testInferenceFailureResetsStateBeforeNextAcceptedAudio() throws {
        let engine = RealtimeTestEngine()
        let stream = try makeStream(engine)
        try stream.submit(block(0))
        _ = try stream.processNext { self.position() }
        try stream.submit(block(320))
        engine.failNextPush = true
        XCTAssertThrowsError(try stream.processNext { self.position() })
        XCTAssertEqual(stream.pendingSampleCount, 0)
        try stream.submit(block(640))
        XCTAssertNil(try stream.processNext { self.position() })
        try stream.submit(block(960))
        let output = try XCTUnwrap(stream.processNext { self.position() })
        XCTAssertEqual(output.startSample, 640)
        XCTAssertEqual(engine.resetCount, 2)
        XCTAssertEqual(stream.statistics.inferenceFailures, 1)
    }

    func testInvalidModelOutputRequiresFreshState() throws {
        for invalid in [[Float](repeating: .infinity, count: 320), [Float](repeating: 1, count: 319)] {
            let engine = RealtimeTestEngine()
            let stream = try makeStream(engine)
            try stream.submit(block(0))
            _ = try stream.processNext { self.position() }
            try stream.submit(block(320))
            engine.outputOverride = invalid
            XCTAssertThrowsError(try stream.processNext { self.position() })
            engine.outputOverride = nil
            try stream.submit(block(640))
            XCTAssertNil(try stream.processNext { self.position() })
            try stream.submit(block(960))
            XCTAssertEqual(try stream.processNext { self.position() }?.startSample, 640)
            XCTAssertEqual(engine.resetCount, 2)
        }
    }

    func testConfigurationRejectsNegativeDelayAndSubHopCapacity() {
        XCTAssertThrowsError(try RealtimeStreamingConfiguration(playoutDelaySamples: -1))
        XCTAssertThrowsError(try RealtimeStreamingConfiguration(playoutDelaySamples: 1_920, maximumPendingSamples: 319))
        XCTAssertNoThrow(try RealtimeStreamingConfiguration(playoutDelaySamples: 0, maximumPendingSamples: 320))
    }
}
