import ReuseAudio
import Foundation

struct ProbeOptions {
    static let usage = """
    Usage: ReuseProbe --input AUDIO --output OUTPUT.wav [options]
      --model PATH  Pinned checkpoint; default REUSE_MODEL_PATH or models/model.safetensors
      --benchmark   Run inference three times; retain the final output
      --optimization MODE  standard (default), specialized-metal, scan-only,
                           normalization-only, prepared-serial, scheduling-only
      --force       Replace an existing output WAV and companion JSON report
      --help, -h    Print this help without loading the model

    Runs on the Apple Silicon GPU using Swift/MLX/Metal. Input: mono/stereo,
    8–192 kHz, up to 60 seconds. Output: 16 kHz mono float WAV plus OUTPUT.json.
    Relative paths are resolved from your current working directory.
    """
    let input: URL
    let output: URL
    let model: URL
    let benchmark: Bool
    let force: Bool
    let optimization: ProbeOptimization
    var report: URL { output.deletingPathExtension().appendingPathExtension("json") }

    init(_ arguments: [String]) throws {
        let parsed = try Self.parse(arguments)
        guard let input = parsed["--input"], let output = parsed["--output"] else {
            throw ReuseError.invalid("Both --input and --output are required.")
        }
        self.input = URL(fileURLWithPath: input).standardizedFileURL
        self.output = URL(fileURLWithPath: output).standardizedFileURL
        let preferred = "models/model.safetensors"
        let legacy = "Resources/model.safetensors"
        let local = FileManager.default.fileExists(atPath: preferred) || !FileManager.default.fileExists(atPath: legacy)
            ? preferred : legacy
        let fallback = ProcessInfo.processInfo.environment["REUSE_MODEL_PATH"] ?? local
        model = URL(fileURLWithPath: parsed["--model"] ?? fallback).standardizedFileURL
        benchmark = parsed["--benchmark"] != nil
        force = parsed["--force"] != nil
        optimization = try ProbeOptimization.parse(parsed["--optimization"] ?? "standard")
    }

    private static func parse(_ arguments: [String]) throws -> [String: String] {
        var result = [String: String](), remaining = ArraySlice(arguments)
        while let argument = remaining.popFirst() {
            guard result[argument] == nil else { throw ReuseError.invalid("Duplicate option: \(argument)") }
            if ["--benchmark", "--force"].contains(argument) { result[argument] = "true"; continue }
            guard ["--input", "--output", "--model", "--optimization"].contains(argument) else {
                throw ReuseError.invalid("Unknown option: \(argument)")
            }
            guard let value = remaining.popFirst(), !value.hasPrefix("--"), !value.isEmpty else {
                throw ReuseError.invalid("Missing value for \(argument).")
            }
            result[argument] = value
        }
        return result
    }

    func validatePaths() throws {
        guard output.pathExtension.lowercased() == "wav" else {
            throw ReuseError.invalid("The output filename must end in .wav.")
        }
        for destination in [output, report] {
            let resolved = destination.resolvingSymlinksInPath()
            guard ![input, model].contains(where: { $0.resolvingSymlinksInPath() == resolved }) else {
                throw ReuseError.invalid("Output paths must differ from input and model paths.")
            }
            try validateDestination(destination)
        }
        for source in [input, model] where !FileManager.default.isReadableFile(atPath: source.path) {
            throw ReuseError.invalid("Cannot read \(source.path).")
        }
    }

    private func validateDestination(_ destination: URL) throws {
        var directory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: destination.path, isDirectory: &directory)
        guard !directory.boolValue, force || !exists else {
            throw ReuseError.invalid("Output exists or is a directory: \(destination.path). Use --force to replace files.")
        }
        guard FileManager.default.isWritableFile(atPath: destination.deletingLastPathComponent().path) else {
            throw ReuseError.invalid("Output directory is missing or not writable.")
        }
    }
}
