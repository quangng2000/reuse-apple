import SwiftUI

struct IntroductionView: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("ON-DEVICE SPEECH RESTORATION", systemImage: "waveform")
                .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            Text("A clearer recording.").font(.largeTitle.bold())
            Text("Enhance short recordings privately on this device. Your audio stays on this device.")
                .foregroundStyle(.secondary)
        }
    }
}
