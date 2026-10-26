import Foundation
import Darwin

let arguments = CommandLine.arguments
guard arguments.count == 4 else {
    fputs("Usage: ReuseStreamingStageProfileMac BENCH_ASSETS_DIRECTORY PROFILE_SOURCE_JSON OUTPUT_DIRECTORY\n", stderr)
    exit(2)
}
do {
    let runner = try EightLayerStageProfile(progress: { print($0) },
        assetsURL: URL(fileURLWithPath: arguments[1], isDirectory: true),
        sourceManifestURL: URL(fileURLWithPath: arguments[2]),
        outputURL: URL(fileURLWithPath: arguments[3], isDirectory: true))
    let count = try runner.run()
    print("Completed \(count) measured diagnostic frames; exact output and full-state gates passed.")
} catch {
    fputs("Stage profile failed: \(error.localizedDescription)\n", stderr)
    exit(1)
}
