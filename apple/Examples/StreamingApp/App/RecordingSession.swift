import AVFoundation
import SwiftUI

@MainActor
final class RecordingSession: ObservableObject {
    @Published var status = "Record up to 20 seconds, then compare both recordings."
    @Published var isRecording = false
    @Published var isBusy = false
    @Published var elapsed = 0.0
    @Published var backlog = 0.0
    @Published var error: String?
    @Published var directory: URL?
    let capture = MicrophoneCapture()
    let worker = RecordingWorker()
    var progress = RecordingProgress()
    var identifier = UUID()
    var timer: Timer?
    var player: AVAudioPlayer?

    func start() {
        guard !isBusy, !isRecording else { return }
        player?.stop(); error = nil; directory = nil
        elapsed = 0; backlog = 0; isBusy = true
        progress = RecordingProgress(); identifier = UUID()
        status = "Preparing the streaming model…"
        let current = identifier
        AVAudioSession.sharedInstance().requestRecordPermission { allowed in
            Task { @MainActor in
                guard self.identifier == current else { return }
                if allowed { self.prepareModel(current) }
                else { self.fail("Allow microphone access in Settings to record.") }
            }
        }
    }

    func stop() {
        guard isRecording else { return }
        do { try capture.finishCapture() } catch { fail(error.localizedDescription); return }
        isRecording = false; isBusy = true
        status = "Finishing queued audio and saving…"
        let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let output = root.appendingPathComponent("recording-\(UUID().uuidString)")
        let current = identifier
        worker.finish(directory: output) { result in
            Task { @MainActor in
                guard self.identifier == current else { return }
                self.completed(result)
            }
        }
    }

    func interrupt() {
        guard isBusy || isRecording else { return }
        identifier = UUID()
        fail("Recording interrupted. Keep the app open during enhancement.")
    }

    func fail(_ message: String) {
        progress.cancel(); capture.stop(); worker.discard()
        timer?.invalidate(); player?.stop()
        isRecording = false; isBusy = false
        error = message; status = "Could not finish the recording."
        UIApplication.shared.isIdleTimerDisabled = false
    }
}
