import Foundation
import ReuseAudio
import XCTest

final class AudioFileTests: XCTestCase {
    func testFloatWAVRoundTripPreservesSamples() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let path = directory.appendingPathComponent("samples.wav")
        let samples: [Float] = (0..<641).map { Float(sin(Double($0) * 0.07) * 0.2) }
        try AudioFileIO.write(samples, to: path)
        XCTAssertEqual(try AudioFileIO.read(path), samples)
    }

    func testInvalidSamplesCannotCreateAudioFiles() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).wav")
        let invalidInputs: [[Float]] = [[], [.nan], [.infinity], [-.infinity]]
        for samples in invalidInputs {
            XCTAssertThrowsError(try AudioFileIO.write(samples, to: path))
            XCTAssertFalse(FileManager.default.fileExists(atPath: path.path))
        }
    }

    func testMissingInputFailsWithoutCreatingAFile() {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).wav")
        XCTAssertThrowsError(try AudioFileIO.read(path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: path.path))
    }
}
