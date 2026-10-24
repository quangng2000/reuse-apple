import Foundation
import MLX
import UIKit

extension EnhancementSession {
    func enhance() {
        guard let inputURL, !busy, !recording, !permissionPending else { return }
        guard let modelURL = Bundle.main.url(forResource: "model", withExtension: "safetensors") else {
            status = "Model missing. Run the model preparation script and rebuild the app."; return
        }
        prepareEnhancement()
        let target = destination("enhanced")
        let operation = operationID
        task = Task.detached(priority: .userInitiated) { [self] in
            await enhanceInBackground(inputURL, model: modelURL, target: target, operation: operation)
        }
    }

    private func prepareEnhancement() {
        stopPlayback()
        busy = true; progress = 0; outputURL = nil; elapsed = nil
        operationID = UUID()
        UIApplication.shared.isIdleTimerDisabled = true
        status = "Loading on-device model…"
    }

    nonisolated private func enhanceInBackground(_ input: URL, model: URL, target: URL, operation: UUID) async {
        do {
            let result = try AudioEnhancement.run(input, model: model, target: target) { fraction in
                Task { @MainActor [weak self] in self?.reportProgress(fraction, operation: operation) }
            }
            await completeEnhancement(target, result: result)
        } catch is CancellationError {
            await reportEnhancementCancellation()
        } catch {
            await reportEnhancementError(error)
        }
        GPU.clearCache()
        await finishTask()
    }

    private func reportProgress(_ fraction: Double, operation: UUID) {
        guard operationID == operation, busy, outputURL == nil, task?.isCancelled == false else { return }
        progress = fraction
        status = "Enhancing on this device · \(Int(fraction * 100))%"
    }

    private func completeEnhancement(_ target: URL, result: EnhancementResult) {
        guard task?.isCancelled == false else { reportEnhancementCancellation(); return }
        outputURL = target
        elapsed = result.elapsed
        duration = result.duration
        progress = 1
        status = result.reducedLevel
            ? "Done. Output level reduced to avoid clipping."
            : "Done. Compare the input and enhanced audio."
    }

    private func reportEnhancementCancellation() {
        status = "Enhancement cancelled. Input preserved."
    }

    private func reportEnhancementError(_ error: Error) {
        status = error.localizedDescription
    }
}
