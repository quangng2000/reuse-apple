import Foundation

final class RecordingProgress {
    private let lock = NSLock()
    private var captured = 0
    private var processed = 0
    private var cancelled = false
    private var maximumBacklog = 0.0
    private var maximumChunk = 0

    func allowance(_ count: Int) -> Int {
        lock.lock(); defer { lock.unlock() }
        guard !cancelled else { return 0 }
        let allowed = max(0, min(count, 320000 - captured))
        captured += allowed
        maximumChunk = max(maximumChunk, allowed)
        maximumBacklog = max(maximumBacklog, Double(max(0, captured - processed - 640)) / 16000)
        return allowed
    }
    func completed(_ count: Int) { lock.lock(); processed = count; lock.unlock() }
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    var isCancelled: Bool { lock.lock(); defer { lock.unlock() }; return cancelled }
    func snapshot() -> (seconds: Double, backlog: Double) {
        lock.lock(); defer { lock.unlock() }
        return (Double(captured) / 16000, Double(max(0, captured - processed - 640)) / 16000)
    }
    func diagnostics() -> (backlog: Double, callback: Double) {
        lock.lock(); defer { lock.unlock() }
        return (maximumBacklog, Double(maximumChunk) / 16000)
    }
}
