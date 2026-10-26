import AVFoundation
import SwiftUI

extension RecordingSession {
    func completed(_ result: Result<URL, Error>) {
        timer?.invalidate()
        UIApplication.shared.isIdleTimerDisabled = false
        isBusy = false
        switch result {
        case .success(let output):
            directory = output; backlog = 0
            status = "Saved. Compare the original and restored audio."
        case .failure(let error): fail(error.localizedDescription)
        }
    }

    func play(_ name: String) {
        guard let directory, !isBusy, !isRecording else { return }
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default)
            try session.setActive(true)
            player = try AVAudioPlayer(contentsOf: directory.appendingPathComponent("play-\(name).wav"))
            player?.play()
        } catch { self.error = error.localizedDescription }
    }
}
