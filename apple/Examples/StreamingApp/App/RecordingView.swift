import SwiftUI

@MainActor
struct RecordingView: View {
    @StateObject private var recording = RecordingSession()
    @Environment(\.scenePhase) private var scenePhase
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    Text("Streaming speech enhancement").font(.largeTitle.bold())
                    Text("Speak into the microphone. The package processes incoming audio on this device; stop to listen to the result.")
                    controls
                    if recording.isRecording || recording.isBusy { progress }
                    Text(recording.status).foregroundStyle(.secondary)
                    if let error = recording.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
                    if recording.directory != nil { playback }
                    Text("Research example · NVIDIA noncommercial model. This is a microphone test; echo cancellation and a connected call are not implemented.")
                        .font(.footnote).foregroundStyle(.secondary)
                    Link("Model and license", destination: URL(string: "https://huggingface.co/nvidia/Real-time_RE-USE#licenseterms-of-use")!)
                }.padding(24)
            }.navigationTitle("RE-USE Streaming").navigationBarTitleDisplayMode(.inline)
        }.onChange(of: scenePhase) { _, phase in
            // The permission alert temporarily makes the scene inactive before capture starts.
            if phase == .background || (phase == .inactive && recording.isRecording) { recording.interrupt() }
        }
    }
    private var controls: some View {
        HStack {
            Button { recording.start() } label: { Label("Record", systemImage: "mic.fill") }
                .buttonStyle(.borderedProminent).disabled(recording.isBusy || recording.isRecording)
            Button { recording.stop() } label: { Label("Stop", systemImage: "stop.fill") }
                .buttonStyle(.bordered).disabled(!recording.isRecording)
        }
    }
    private var progress: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(String(format: "Recorded %.1f seconds", recording.elapsed)).monospacedDigit()
            Text(String(format: "Processing backlog %.2f seconds", recording.backlog)).monospacedDigit()
            if recording.isBusy { ProgressView() }
        }
    }
    private var playback: some View {
        HStack {
            Button("Play original") { recording.play("original") }
            Button("Play restored") { recording.play("restored") }
        }.buttonStyle(.bordered).disabled(recording.isBusy || recording.isRecording)
    }
}
