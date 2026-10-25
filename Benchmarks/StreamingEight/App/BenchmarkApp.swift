import SwiftUI
import UIKit

@MainActor
final class BenchmarkViewModel: ObservableObject {
    @Published var status = "Ready for the eight-layer FP32 comparison."
    @Published var running = false

    func run(profileOnly: Bool = false) {
        guard !running else { return }
        running = true
        UIApplication.shared.isIdleTimerDisabled = true
        UIDevice.current.isBatteryMonitoringEnabled = true
        status = profileOnly ? "Loading pinned stage-profile diagnostic…" : "Loading frozen baseline and candidate…"
        Task.detached(priority: .userInitiated) { [self] in
            let runID = UUID().uuidString
            do {
                let progress: (String) -> Void = { message in
                    Task { @MainActor [self] in self.status = message }
                }
                let message: String
                if profileOnly {
                    let runner = try EightLayerStageProfile(progress: progress)
                    let frames = try runner.run()
                    message = "Profile complete. \(frames) measured frames saved in Documents/latest-profile.json."
                } else {
                    let runner = try EightLayerBenchmark(progress: progress, runID: runID)
                    let report = try runner.run()
                    message = "Complete. \(report) trials, run \(runID), saved in Documents/latest-results.json."
                }
                await MainActor.run {
                    self.status = message
                }
            } catch {
                if profileOnly { EightLayerStageProfile.recordFailure(error) }
                else { EightLayerBenchmark.recordFailure(error, runID: runID) }
                await MainActor.run { self.status = "Stopped: " + error.localizedDescription }
            }
            await MainActor.run {
                self.running = false
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }
}

@main
struct ReuseStreamingEightBenchmarkApp: App {
    @StateObject private var model = BenchmarkViewModel()
    var body: some Scene {
        WindowGroup {
            VStack(alignment: .leading, spacing: 18) {
                Text("Eight-layer FP32 benchmark").font(.title2)
                Text("Current optimized implementation versus two new optimizations. Both use all eight layers and the same checkpoint.")
                Text(model.status).font(.callout).monospacedDigit()
                Button(model.running ? "Running…" : "Run comparison") { model.run() }
                    .disabled(model.running)
                Text("Local file replay, not microphone or call latency. Noncommercial research model.")
                    .font(.footnote).foregroundStyle(.secondary)
                Spacer()
            }
            .padding(24)
            .task {
                let arguments = ProcessInfo.processInfo.arguments
                if arguments.contains("--profile-only") { model.run(profileOnly: true) }
                else if arguments.contains("--autorun") { model.run() }
            }
        }
    }
}
