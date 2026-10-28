import XCTest
@testable import ReuseStreaming

/// Virtual 20 ms input/render callbacks continue while the synchronous model is busy.
/// A bounded host ingress queue is refreshed before every one-hop processNext call.
private final class RealtimeSimulation {
    let delay: UInt64 = 1_920
    var clock = 0
    var nextCallback = 0
    var nextSample: UInt64 = 0
    var ingressStart: UInt64 = 0
    var ingress = [Float]()
    var output = [UInt64: Float]()
    var wetBySecond = [Int: Int]()
    var misalignedSamples = 0
    var usedStall = false

    func samples(at start: UInt64, count: Int = 320) -> [Float] {
        (0..<count).map { Float((start + UInt64($0)) % 5_000 + 1) / 10_000 }
    }

    func advance(to end: Int) {
        while nextCallback <= end {
            let start = nextSample
            if ingress.isEmpty { ingressStart = start }
            ingress += samples(at: start)
            // The host's queue is bounded too; overload can create an input gap.
            if ingress.count > 4_096 {
                let dropped = ingress.count - 4_096
                ingress.removeFirst(dropped)
                ingressStart += UInt64(dropped)
            }
            for offset in 0..<320 {
                let rendered = start + UInt64(offset)
                if rendered < delay { continue }
                let source = rendered - delay
                if let value = output.removeValue(forKey: source) {
                    let expected = Float(source % 5_000 + 1) / 10_000 * 0.5
                    if value != expected { misalignedSamples += 1 }
                    wetBySecond[nextCallback / 1_000, default: 0] += 1
                }
            }
            nextSample += 320
            nextCallback += 20
        }
        clock = end
    }

    func run(warm: Int, cold: Int, stall: Bool) throws -> (RealtimeStreamingStatistics, Int) {
        let engine = RealtimeTestEngine()
        let stream = RealtimeStreamingEnhancer(
            configuration: try RealtimeStreamingConfiguration(playoutDelaySamples: Int(delay)),
            push: engine.push, reset: engine.reset)
        engine.onPush = {
            // Match the core: first push has no frame, second does cold+warm frames.
            let duration = engine.pushesSinceReset == 1 ? 0 : warm + (engine.pushesSinceReset == 2 ? cold : 0)
            let extra = stall && !self.usedStall && self.clock >= 500 ? 700 : 0
            if extra > 0 { self.usedStall = true }
            self.advance(to: self.clock + duration + extra)
        }
        advance(to: 0)
        while clock < 8_000 {
            if !ingress.isEmpty {
                try stream.submit(RealtimeAudioBlock(samples: ingress, startSample: ingressStart, generation: 1))
                ingress.removeAll(keepingCapacity: true)
            }
            let pushes = engine.pushCount
            let restored = try stream.processNext {
                RealtimePlaybackPosition(nextSample: self.nextSample, generation: 1)
            }
            XCTAssertLessThanOrEqual(engine.pushCount - pushes, 1)
            XCTAssertLessThanOrEqual(stream.pendingSampleCount, 3_200)
            if let restored {
                XCTAssertEqual(restored.generation, 1)
                for (index, value) in restored.samples.enumerated() {
                    output[restored.startSample + UInt64(index)] = value
                }
            }
            advance(to: clock + 2)
        }
        engine.onPush = nil
        return (stream.statistics, engine.resetCount)
    }
}

extension RealtimeStreamingTests {
    func testHealthyVirtualCallHasContinuousOutputAfterStartup() throws {
        let simulation = RealtimeSimulation()
        let (statistics, resets) = try simulation.run(warm: 17, cold: 26, stall: false)
        for second in 2...7 { XCTAssertEqual(simulation.wetBySecond[second], 16_000) }
        XCTAssertEqual(simulation.misalignedSamples, 0)
        XCTAssertEqual(statistics.deadlineResets, 0)
        XCTAssertEqual(resets, 1)
    }

    func testLongStallRecoversWithoutPermanentlyDiscardingAllOutput() throws {
        let simulation = RealtimeSimulation()
        let (statistics, _) = try simulation.run(warm: 17, cold: 26, stall: true)
        XCTAssertTrue(simulation.usedStall)
        for second in 3...7 { XCTAssertGreaterThanOrEqual(simulation.wetBySecond[second, default: 0], 15_000) }
        XCTAssertEqual(simulation.misalignedSamples, 0)
        XCTAssertGreaterThan(statistics.discardedInputSamples, 0)
    }

    func testSustainedOverloadStillRejoinsFreshAudioInsteadOfResetLoop() throws {
        let simulation = RealtimeSimulation()
        let (statistics, _) = try simulation.run(warm: 34, cold: 45, stall: false)
        for second in 2...7 {
            XCTAssertGreaterThan(simulation.wetBySecond[second, default: 0], 0, "No fresh output in second \(second)")
        }
        XCTAssertEqual(simulation.misalignedSamples, 0)
        XCTAssertGreaterThan(statistics.deadlineResets, 0)
        XCTAssertGreaterThan(statistics.discardedInputSamples, 0)
        // No claim of continuous playback: 34 ms inference cannot sustain a 20 ms cadence.
    }
}
