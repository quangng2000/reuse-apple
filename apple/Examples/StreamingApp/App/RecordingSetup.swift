import AVFoundation
import SwiftUI

extension RecordingSession {
    func prepareModel(_ current: UUID) {
        guard let model = Bundle.main.url(forResource: "realtime-reuse", withExtension: "safetensors") else {
            fail("Run apple/scripts/prepare_streaming_model.py, then rebuild this example."); return
        }
        UIApplication.shared.isIdleTimerDisabled = true
        worker.prepare(modelURL: model, progress: progress) { result in
            Task { @MainActor in
                guard self.identifier == current else { return }
                switch result {
                case .success: self.beginCapture(current)
                case .failure(let error): self.fail(error.localizedDescription)
                }
            }
        }
    }

    func beginCapture(_ current: UUID) {
        guard UIApplication.shared.applicationState == .active else { interrupt(); return }
        let progress = self.progress, worker = self.worker
        do {
            try capture.start(onSamples: { samples in
                let allowed = progress.allowance(samples.count)
                if allowed > 0 { worker.append(Array(samples.prefix(allowed))) }
            }, onFailure: { message in
                Task { @MainActor in
                    guard self.identifier == current else { return }
                    self.error = message; self.stop()
                }
            })
            isBusy = false; isRecording = true
            status = "Recording with four-layer streaming enhancement."
            startTimer()
        } catch { fail(error.localizedDescription) }
    }

    private func startTimer() {
        timer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                let snapshot = self.progress.snapshot()
                self.elapsed = snapshot.seconds; self.backlog = snapshot.backlog
                if self.elapsed >= 20, self.isRecording { self.stop() }
            }
        }
    }
}
