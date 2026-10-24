import SwiftUI

struct EnhancementControls: View {
    @ObservedObject var session: EnhancementSession

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(session.status).font(.callout).accessibilityIdentifier("status")
            processingControls
            if let input = session.inputURL { PlaybackButton(title: "Original", url: input, session: session) }
            if let output = session.outputURL {
                PlaybackButton(title: "Enhanced", url: output, session: session)
                ShareLink(item: output) { Label("Save or share WAV", systemImage: "square.and.arrow.up") }
            }
            timing
        }.padding(20).frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 20))
    }

    @ViewBuilder private var processingControls: some View {
        if session.busy {
            ProgressView(value: session.progress)
            Button("Cancel", role: .cancel, action: session.cancel)
        } else if session.inputURL != nil {
            Button(action: session.enhance) {
                Label("Enhance recording", systemImage: "waveform.badge.magnifyingglass")
                    .frame(maxWidth: .infinity)
            }.buttonStyle(.borderedProminent).tint(.teal).disabled(session.recording)
        }
    }

    @ViewBuilder private var timing: some View {
        if let seconds = session.elapsed, let duration = session.duration {
            Text(String(format: "%.1f s to process %.1f s of audio", seconds, duration))
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
    }
}
