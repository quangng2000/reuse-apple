import AVFoundation
import UIKit

extension EnhancementSession {
    func toggleRecord() {
        if recording { finishRecording(); return }
        guard !busy, !permissionPending else { return }
        permissionPending = true
        Task { await requestRecording() }
    }

    private func requestRecording() async {
        let granted = await AVAudioApplication.requestRecordPermission()
        permissionPending = false
        guard granted else { status = "Allow microphone access in Settings to record audio."; return }
        guard UIApplication.shared.applicationState == .active else { return }
        do { try startRecording() }
        catch { status = error.localizedDescription; recording = false }
    }

    private func startRecording() throws {
        stopPlayback()
        try AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try AVAudioSession.sharedInstance().setActive(true)
        recorder = try AVAudioRecorder(url: destination("recording"), settings: recordingSettings)
        recorder?.delegate = self
        guard recorder?.record(forDuration: 30) == true else { throw ReuseError.invalid("Could not start recording.") }
        inputURL = nil; outputURL = nil; elapsed = nil; duration = nil
        recording = true
        UIApplication.shared.isIdleTimerDisabled = true
        status = "Recording · stops automatically after 30 seconds."
    }

    private var recordingSettings: [String: Any] {
        [AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 48000,
         AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
         AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false]
    }

    func finishRecording() {
        guard let recorder else { return }
        let url = recorder.url
        let seconds = recorder.currentTime
        recorder.stop()
        self.recorder = nil
        recording = false
        UIApplication.shared.isIdleTimerDisabled = false
        if seconds >= 0.1 {
            inputURL = url; duration = seconds; status = "Recording saved. Ready to enhance."
        } else { inputURL = nil; status = "Record at least a fraction of a second." }
    }
}

extension EnhancementSession: AVAudioRecorderDelegate {
    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        let url = recorder.url
        Task { @MainActor [weak self] in
            guard let self, self.recording, self.recorder?.url == url else { return }
            self.recording = false
            UIApplication.shared.isIdleTimerDisabled = false
            self.recorder = nil
            if flag {
                self.inputURL = url; self.duration = 30
                self.status = "Recording saved. Ready to enhance."
            } else { self.status = "Recording was interrupted." }
        }
    }
}
