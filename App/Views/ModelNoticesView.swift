import SwiftUI

struct ModelNoticesView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView { noticesContent.padding() }
                .toolbar { Button("Done") { dismiss() } }
        }
    }

    private var noticesContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("NVIDIA NemotronLabs RE-USE").font(.title2.bold())
            Text("OpenMDW-1.1 checkpoint\n30 layers · 9.61 million parameters\nSwift / MLX / Metal prototype")
            Link("Official model card", destination: URL(string: "https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE")!)
            Text(notices).font(.caption).textSelection(.enabled)
        }
    }

    private var notices: String {
        ["MODEL-LICENSE", "THIRD-PARTY-NOTICES"].map(readNotice).joined(separator: "\n\n")
    }

    private func readNotice(_ name: String) -> String {
        guard let url = Bundle.main.url(forResource: name, withExtension: "txt") else {
            return "\(name) is missing. Rebuild with the notice resources included."
        }
        do { return try String(contentsOf: url, encoding: .utf8) }
        catch { return "Could not read \(name): \(error.localizedDescription)" }
    }
}
