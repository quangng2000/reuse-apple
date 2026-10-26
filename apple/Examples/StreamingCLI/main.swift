import Foundation

do {
    let arguments = Array(CommandLine.arguments.dropFirst())
    if arguments.contains("--help") || arguments.contains("-h") {
        print(StreamingOptions.usage)
    } else {
        try StreamingCommand.run(StreamingOptions(arguments))
    }
} catch {
    fputs("ReuseStreamingCLI: \(error.localizedDescription)\n", stderr)
    exit(1)
}
