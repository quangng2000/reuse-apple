import Foundation
import ReuseAudio

final class RecordingWorker {
    private let queue = DispatchQueue(label: "io.github.quangng2000.reuse-streaming.worker", qos: .userInitiated)
    private var processor: RecordingProcessor?
    private var failure: Error?

    func prepare(modelURL: URL, progress: RecordingProgress, completion: @escaping (Result<Void, Error>) -> Void) {
        queue.async {
            self.failure = nil
            do {
                self.processor = try RecordingProcessor(modelURL: modelURL, progress: progress)
                completion(.success(()))
            } catch { self.processor = nil; completion(.failure(error)) }
        }
    }
    func append(_ samples: [Float]) {
        queue.async {
            guard self.failure == nil else { return }
            do { try self.processor?.append(samples) } catch { self.failure = error }
        }
    }
    func discard() { queue.async { self.processor = nil; self.failure = nil } }
    func finish(directory: URL, completion: @escaping (Result<URL, Error>) -> Void) {
        queue.async {
            defer { self.processor = nil; self.failure = nil }
            do {
                if let failure = self.failure { throw failure }
                guard let processor = self.processor else { throw ReuseError.invalid("The model is not ready.") }
                completion(.success(try processor.finish(directory: directory)))
            } catch { completion(.failure(error)) }
        }
    }
}
