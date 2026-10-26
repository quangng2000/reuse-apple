import Foundation
import ReuseAudio

enum StreamingOutput {
    static func save(_ options: StreamingOptions, rendered: RenderedAudio, timing: RunTiming,
                     optimized: Bool) throws {
        try options.validatePaths()
        let temporary = options.output.deletingLastPathComponent().appendingPathComponent(".reuse-streaming-\(UUID()).wav")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try AudioFileIO.write(rendered.samples, to: temporary)
        let report = try StreamingReport.data(options, rendered: rendered, timing: timing,
                                               optimized: optimized, outputURL: temporary)
        // Recheck after inference and temporary-file construction, before committing anything.
        try options.validatePaths()
        try publishAudio(temporary, options: options)
        try report.write(to: options.report, options: options.force ? .atomic : .withoutOverwriting)
    }

    private static func publishAudio(_ temporary: URL, options: StreamingOptions) throws {
        if options.force {
            guard rename(temporary.path, options.output.path) == 0 else {
                throw ReuseError.invalid("Cannot replace output: \(String(cString: strerror(errno)))")
            }
        } else {
            // moveItem refuses an existing destination, including a newly created symlink.
            try FileManager.default.moveItem(at: temporary, to: options.output)
        }
    }
}
