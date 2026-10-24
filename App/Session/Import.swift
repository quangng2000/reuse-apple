import Foundation

extension EnhancementSession {
    func importAudio(_ url: URL) {
        guard !busy, !recording, !permissionPending else { return }
        stopPlayback()
        busy = true
        operationID = UUID()
        progress = 0
        status = "Reading audio…"
        let target = destination("input")
        task = Task.detached(priority: .userInitiated) { [self] in
            await importInBackground(url, target: target)
        }
    }

    nonisolated private func importInBackground(_ url: URL, target: URL) async {
        do {
            let samples = try Self.readImport(url, target: target)
            await completeImport(target, sampleCount: samples.count)
        } catch is CancellationError {
            await setImportCancellation()
        } catch {
            await reportImportError(error)
        }
        await finishTask()
    }

    nonisolated private static func readImport(_ url: URL, target: URL) throws -> [Float] {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        try Task.checkCancellation()
        let samples = try AudioFileIO.read(url)
        try Task.checkCancellation()
        try AudioFileIO.write(samples, to: target)
        try Task.checkCancellation()
        return samples
    }

    private func completeImport(_ target: URL, sampleCount: Int) {
        guard task?.isCancelled == false else { setImportCancellation(); return }
        inputURL = target
        outputURL = nil
        elapsed = nil
        duration = Double(sampleCount) / 16000
        status = "Audio loaded. Ready to enhance."
    }

    private func setImportCancellation() {
        status = "Import cancelled. Previous recording preserved."
    }

    private func reportImportError(_ error: Error) {
        status = error.localizedDescription
    }
}
