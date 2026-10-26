import SwiftUI

struct RecordingControls: View {
    @ObservedObject var session: EnhancementSession
    @Binding var importing: Bool

    var body: some View {
        VStack(spacing: 12) {
            recordButton
            Button { importing = true } label: { Label("Import audio", systemImage: "folder") }
                .buttonStyle(.bordered)
                .disabled(session.busy || session.recording || session.permissionPending)
            Text("Record up to 30 seconds · import up to 60 seconds")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var recordButton: some View {
        Button(action: session.toggleRecord) {
            Label(session.recording ? "Stop recording" : "Record audio",
                  systemImage: session.recording ? "stop.circle.fill" : "mic.fill")
                .frame(maxWidth: .infinity).padding(.vertical, 6)
        }.buttonStyle(.borderedProminent).tint(session.recording ? .red : .teal)
            .disabled(session.busy || session.permissionPending)
    }
}
