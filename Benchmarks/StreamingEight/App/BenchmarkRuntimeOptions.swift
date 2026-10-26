import Foundation

struct BenchmarkRuntimeOptions {
    let candidateCompilation: String?
    let candidatePooling: Bool?
    let shortOnly: Bool
    let recognizedArguments: [String]

    enum Invalid: LocalizedError {
        case argument(String)
        var errorDescription: String? { if case .argument(let message) = self { return message }; return nil }
    }

    init(arguments: [String]) throws {
        var compilation: String?, pooling: Bool?, short = false
        var recognized = [String]()
        var index = 1
        while index < arguments.count {
            let argument = arguments[index]
            switch argument {
            case "--candidate-compilation":
                guard compilation == nil, index + 1 < arguments.count,
                      ["disabled", "body", "decoderFeatures"].contains(arguments[index + 1]) else {
                    throw Invalid.argument("--candidate-compilation requires one of disabled, body, decoderFeatures exactly once")
                }
                compilation = arguments[index + 1]
                recognized += [argument, arguments[index + 1]]
                index += 1
            case "--candidate-pooling":
                guard pooling == nil, index + 1 < arguments.count,
                      ["true", "false"].contains(arguments[index + 1]) else {
                    throw Invalid.argument("--candidate-pooling requires true or false exactly once")
                }
                pooling = arguments[index + 1] == "true"
                recognized += [argument, arguments[index + 1]]
                index += 1
            case "--short-only":
                guard !short, index + 1 == arguments.count || !["true", "false"].contains(arguments[index + 1]) else {
                    throw Invalid.argument("--short-only is a flag without a value and may appear only once")
                }
                short = true
                recognized.append(argument)
            default:
                if argument.hasPrefix("--candidate") || argument.hasPrefix("--short") {
                    throw Invalid.argument("Unknown benchmark runtime argument: " + argument)
                }
                // Other arguments belong to iOS/Xcode or the app's launch mode.
                break
            }
            index += 1
        }
        candidateCompilation = compilation
        candidatePooling = pooling
        shortOnly = short
        recognizedArguments = recognized
    }
}
