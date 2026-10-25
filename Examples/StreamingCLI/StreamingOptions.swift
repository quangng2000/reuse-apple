import Foundation
import ReuseAudio

struct StreamingOptions {
    static let usage = """
    ReuseStreamingCLI --input AUDIO --output OUTPUT.wav [options]
      --model PATH  Pinned model; default Resources/realtime-reuse.safetensors
      --layers N    Streaming exit 3–12; default 4, zero lookahead
      --benchmark   Save per-frame timing, warmup, and loading statistics
      --baseline    Disable optimized kernels for numerical comparison
      --force       Replace existing output WAV and companion JSON
      --help, -h    Show help without loading a checkpoint

    Noncommercial research model. Audio is processed locally on the Apple GPU.
    Input: mono/stereo audio up to 60 seconds. Output: 16 kHz mono float WAV.
    This replays a file as fast as possible; it is not a live-call latency test.
    """
    let input: URL
    let output: URL
    let model: URL
    let layers: Int
    let benchmark: Bool
    let baseline: Bool
    let force: Bool
    var report: URL { output.deletingPathExtension().appendingPathExtension("json") }

    init(_ arguments: [String]) throws {
        let options = try Self.parse(arguments)
        guard let input = options["--input"], let output = options["--output"],
              let layers = Int(options["--layers"] ?? "4"), (3...12).contains(layers) else {
            throw ReuseError.invalid("Supply --input and --output, with --layers between 3 and 12.")
        }
        self.input = URL(fileURLWithPath: input).standardizedFileURL
        self.output = URL(fileURLWithPath: output).standardizedFileURL
        model = URL(fileURLWithPath: options["--model"] ?? "Resources/realtime-reuse.safetensors").standardizedFileURL
        self.layers = layers
        benchmark = options["--benchmark"] != nil
        baseline = options["--baseline"] != nil
        force = options["--force"] != nil
    }

    private static func parse(_ arguments: [String]) throws -> [String: String] {
        var result = [String: String](), remaining = ArraySlice(arguments)
        while let option = remaining.popFirst() {
            guard result[option] == nil else { throw ReuseError.invalid("Duplicate option: \(option)") }
            if ["--benchmark", "--baseline", "--force"].contains(option) { result[option] = "true"; continue }
            guard ["--input", "--output", "--model", "--layers"].contains(option),
                  let value = remaining.popFirst(), !value.hasPrefix("--"), !value.isEmpty else {
                throw ReuseError.invalid("Unknown option or missing value: \(option)")
            }
            result[option] = value
        }
        return result
    }

    func validatePaths() throws {
        guard output.pathExtension.lowercased() == "wav" else { throw ReuseError.invalid("Output must end in .wav.") }
        for destination in [output, report] {
            let resolved = destination.resolvingSymlinksInPath()
            guard ![input, model].contains(where: { $0.resolvingSymlinksInPath() == resolved }) else {
                throw ReuseError.invalid("Output and report must differ from input and model.")
            }
            try validateDestination(destination)
        }
        for source in [input, model] where !FileManager.default.isReadableFile(atPath: source.path) {
            throw ReuseError.invalid("Cannot read \(source.path).")
        }
    }

    private func validateDestination(_ url: URL) throws {
        var directory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: url.path, isDirectory: &directory)
        guard !directory.boolValue, force || !exists else {
            throw ReuseError.invalid("Output exists: \(url.path). Use --force to replace files.")
        }
        guard FileManager.default.isWritableFile(atPath: url.deletingLastPathComponent().path) else {
            throw ReuseError.invalid("Output directory is missing or not writable.")
        }
    }
}
