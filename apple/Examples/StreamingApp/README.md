# iPhone streaming example

This small SwiftUI app consumes the standalone `ReuseStreaming` and `ReuseAudio`
products through a local Swift Package dependency. It contains no model internals,
purchase logic, personal recordings, or signing credentials.

Install XcodeGen (`brew install xcodegen`), then run from the repository root:

```sh
python3 apple/scripts/prepare_streaming_model.py
xcodegen generate --spec apple/Examples/StreamingApp/project.yml
open apple/Examples/StreamingApp/ReuseStreamingExample.xcodeproj
```

Select the **ReuseStreamingExample** scheme and your physical iPhone, choose your
own development team and unique bundle identifier, then build a **Release**
configuration for performance measurements. An Apple GPU and iOS 17+ are required;
the simulator is not a substitute for device measurements.

Tap **Record**, speak for up to 20 seconds, and tap **Stop**. The serial worker
passes arriving 16 kHz mono samples through `StreamingEnhancer.push`, then flushes
with `finish`. Playback uses one shared peak-safe gain so levels remain comparable.
Original/restored WAVs and a JSON timing report are saved under app Documents and
can be copied through Xcode or Finder file sharing. Keep the app in the foreground.
Normal Stop drains the microphone resampler before flushing the model; cancellation
discards pending audio. Check resampling partition/tail behavior without microphone
permission with `bash apple/Examples/StreamingApp/Checks/run_conversion_check.sh`.

The app requests 20 ms microphone callbacks, but iOS may deliver larger batches.
Watch processing backlog as well as frame timing. This is not a connected call:
there is no acoustic echo cancellation, WebRTC, network, or live speaker output.

NVIDIA's real-time checkpoint is **noncommercial research only**. The complete
license is bundled in the app. See [streaming API and licensing](../../../docs/streaming.md).
