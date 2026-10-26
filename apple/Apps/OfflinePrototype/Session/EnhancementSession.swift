import AVFoundation
import Combine
import Foundation
import UIKit

@MainActor final class EnhancementSession: NSObject, ObservableObject {
    @Published var busy = false
    @Published var recording = false
    @Published var permissionPending = false
    @Published var progress = 0.0
    @Published var status = "Record a short clip or import audio."
    @Published var inputURL: URL?
    @Published var outputURL: URL?
    @Published var playingURL: URL?
    @Published var elapsed: Double?
    @Published var duration: Double?
    var recorder: AVAudioRecorder?
    var player: AVAudioPlayer?
    var task: Task<Void, Never>?
    var operationID = UUID()

    func destination(_ name: String) -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("\(name)-\(UUID().uuidString.prefix(8)).wav")
    }

    func cancel() {
        task?.cancel()
        status = "Cancelling…"
    }

    func finishTask() {
        busy = false
        task = nil
        UIApplication.shared.isIdleTimerDisabled = false
    }

    func leaveForeground() {
        if busy { cancel() }
        if recording { finishRecording() }
        stopPlayback()
    }
}
