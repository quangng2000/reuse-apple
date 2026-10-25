import SwiftUI
import UIKit

@MainActor
final class BenchmarkViewModel: ObservableObject {
    @Published var status = "Ready for the eight-layer FP32 comparison."
    @Published var running = false

    func run() {
        guard !running else { return }
        running = true
        UIApplication.shared.isIdleTimerDisabled = true
        UIDevice.current.isBatteryMonitoringEnabled = true
        status = "Loading frozen baseline and candidate…"
        Task.detached(priority: .userInitiated) { [self] in
            do {
                let runner = try EightLayerBenchmark { message in
                    Task { @MainActor [self] in self.status = message }
                }
                let report = try runner.run()
                await MainActor.run {
                    self.status = "Complete. \(report) trial results saved in Documents/latest-results.json."
                }
            } catch {
                EightLayerBenchmark.recordFailure(error)
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
                if ProcessInfo.processInfo.arguments.contains("--autorun") { model.run() }
            }
        }
    }
}
