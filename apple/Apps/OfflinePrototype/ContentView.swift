import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var session = EnhancementSession()
    @State private var importing = false
    @State private var showingLicense = false
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            content
                .navigationTitle("RE-USE Pocket")
                .navigationBarTitleDisplayMode(.inline)
                .fileImporter(isPresented: $importing, allowedContentTypes: [.audio],
                              onCompletion: handleImport)
                .sheet(isPresented: $showingLicense) { ModelNoticesView() }
                .onChange(of: scenePhase) { _, phase in
                    if phase == .background { session.leaveForeground() }
                }
        }
    }

    private var content: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                IntroductionView()
                RecordingControls(session: session, importing: $importing)
                EnhancementControls(session: session)
                Text("Prototype for saved recordings. Keep the app open while enhancing. Severely damaged speech may not be recoverable.")
                    .font(.footnote).foregroundStyle(.secondary)
                Button("Model & open-source notices") { showingLicense = true }
                    .font(.footnote)
            }.padding(24)
        }
    }

    private func handleImport(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url): session.importAudio(url)
        case .failure(let error): session.status = error.localizedDescription
        }
    }
}
