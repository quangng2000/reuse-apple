// Independent offline regression tests. SPDX-License-Identifier: OpenMDW-1.1
import MLX
import XCTest
@testable import ReuseAudio

final class OfflineSelectiveScanTests: XCTestCase {
    func testSpecializedScanMatchesScalarEquationsForMultipleIndependentBatches() {
        compareScalar { OfflineSelectiveScan() }
    }

    func testPreparedSerialScanMatchesScalarEquationsForMultipleIndependentBatches() {
        compareScalar { OfflineSelectiveScan(execution: .serial) }
    }

    private func compareScalar(_ makeScan: () -> OfflineSelectiveScan) {
        Device.withDefaultDevice(.gpu) {
            let scan = makeScan()
            for (batch, length) in [(1, 1), (2, 7)] {
                let fixture = OfflineScanFixture(batch: batch, length: length)
                let actual = scan(fixture.arrays).asArray(Float.self)
                OfflineMetalTestSupport.assertClose(actual, OfflineScanOracle.evaluate(fixture))
            }
        }
    }

    func testSpecializedScanPreservesSerialRecurrenceAtShortAndModelSequenceLengths() {
        compareSequenceLengths { OfflineSelectiveScan() }
    }

    func testPreparedSerialScanPreservesRecurrenceAtShortAndModelSequenceLengths() {
        compareSequenceLengths { OfflineSelectiveScan(execution: .serial) }
    }

    private func compareSequenceLengths(_ makeScan: () -> OfflineSelectiveScan) {
        Device.withDefaultDevice(.gpu) {
            let scan = makeScan()
            for (batch, length) in [(3, 31), (2, 161), (1, 201)] {
                let arrays = OfflineScanFixture(batch: batch, length: length).arrays
                let expected = SelectiveScan()(arrays[0], arrays[1], arrays[2], arrays[3],
                                                arrays[4], arrays[5], arrays[6], arrays[7])
                OfflineMetalTestSupport.assertClose(scan(arrays), expected)
            }
        }
    }

    func testScanRestartsStateForEachIndependentCall() {
        compareIndependentCalls { OfflineSelectiveScan() }
    }

    func testPreparedSerialScanRestartsStateForEachIndependentCall() {
        compareIndependentCalls { OfflineSelectiveScan(execution: .serial) }
    }

    private func compareIndependentCalls(_ makeScan: () -> OfflineSelectiveScan) {
        Device.withDefaultDevice(.gpu) {
            let scan = makeScan()
            let first = OfflineScanFixture(batch: 2, length: 17).arrays
            let expected = scan(first).asArray(Float.self)
            eval(scan(OfflineScanFixture(batch: 1, length: 3).arrays))
            OfflineMetalTestSupport.assertExact(scan(first).asArray(Float.self), expected)
        }
    }
}
