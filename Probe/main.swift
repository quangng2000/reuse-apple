import Foundation
import MLX

do {
    let arguments = Array(CommandLine.arguments.dropFirst())
    if arguments.contains("--help") || arguments.contains("-h") {
        print(ProbeOptions.usage)
    } else {
        let options = try ProbeOptions(arguments)
        try options.validatePaths()
        try Device.withDefaultDevice(.gpu) { try ProbeRunner.run(options) }
    }
} catch {
    fputs("RE-USE: \(error.localizedDescription)\nUse --help for usage.\n", stderr)
    exit(1)
}
