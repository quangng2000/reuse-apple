import SwiftUI

struct PlaybackButton: View {
    let title: String
    let url: URL
    @ObservedObject var session: EnhancementSession

    var body: some View {
        Button { session.play(url) } label: {
            Label(session.playingURL == url ? "Stop \(title.lowercased())" : "Play \(title.lowercased())",
                  systemImage: session.playingURL == url ? "stop.fill" : "play.fill")
                .frame(maxWidth: .infinity, alignment: .leading)
        }.buttonStyle(.bordered).disabled(session.busy || session.recording)
    }
}
