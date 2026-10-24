import AVFoundation

extension EnhancementSession {
    func stopPlayback() {
        player?.stop()
        player = nil
        playingURL = nil
    }

    func play(_ url: URL) {
        if playingURL == url { stopPlayback(); return }
        guard !busy, !recording else { return }
        do {
            stopPlayback()
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            player = try AVAudioPlayer(contentsOf: url)
            player?.delegate = self
            if player?.play() == true { playingURL = url }
        } catch { status = error.localizedDescription }
    }
}

extension EnhancementSession: AVAudioPlayerDelegate {
    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        let url = player.url
        Task { @MainActor [weak self] in
            guard let self, self.player?.url == url else { return }
            self.player = nil
            self.playingURL = nil
        }
    }
}
