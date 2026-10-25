# Build and check

`Package.swift` defines the reusable `ReuseAudio` and `ReuseStreaming` libraries
and their Mac executables. Add it directly to Xcode projects or Apple-platform
packages; see [package integration](package.md). Final GPU builds use Xcode,
because plain SwiftPM command-line builds do not compile MLX's Metal shaders.

For the streaming tool and iPhone example:

```sh
python3 Scripts/prepare_streaming_model.py
bash Scripts/run_streaming.sh --help
```

The iPhone research example has its own [setup instructions](../Examples/StreamingApp/README.md).
Use a physical Apple device for GPU measurements, Release configuration, and
report loading/warmup separately from frame processing.

## Original offline tool and app

The offline example's generated Xcode project is in `Examples/OfflineApp/`;
XcodeGen is only needed when changing its definition. Swift package revisions
are locked in `Package.resolved`. The repository root is the standalone package.

```sh
python3 Scripts/prepare_model.py
bash Scripts/run_mac.sh --help
python3 Scripts/smoke_test.py
python3 Scripts/boundary_smoke.py
```

The smoke test creates a synthetic chirp, runs one actual GPU inference, and
checks sample rate, sample count, finite audio, input preservation, overwrite
protection, and invalid command-line arguments. It uses Python's standard
library and does not require private recordings or PyTorch.

The boundary checks cover an 80-sample input, silence, stereo 48 kHz conversion,
and a 4.1-second recording spanning two inference windows.

For an unsigned iPhone/iPad device build:

```sh
xcodebuild build -project Examples/OfflineApp/ReusePocket.xcodeproj -scheme ReusePocket \
  -configuration Release -destination 'generic/platform=iOS' \
  -derivedDataPath BuildIOS -clonedSourcePackagesDirPath SourcePackages \
  CODE_SIGNING_ALLOWED=NO
```

An unsigned compile does not install the app. Device testing requires Xcode
signing with your own team and connected hardware. No signing credentials are
included in this repository.

After adding or moving Swift sources, regenerate the project:

```sh
xcodegen generate --spec Examples/OfflineApp/project.yml
```

The first dependency build can take several minutes. Build outputs, model
weights, logs, and recordings stay outside Git. App inference and the command-line
tool share the same engine; changes should preserve numerical agreement with the
pinned reference. Benchmark with `--benchmark` on your own recordings before
making device-speed claims.

## Packaged build checks (2026-10-23)

- ARM64 Release builds passed for the Mac CLI and unsigned iPhone/iPad target.
- Synthetic GPU smoke and all four boundary checks passed on Apple M4 Pro.
- The packaged CLI produced byte-for-byte identical speech output to the validated
  Swift/Metal prototype, with STOI 0.880594 on the same diagnostic clip.
- Three inference passes took 2.348, 2.307, and 2.312 seconds for four seconds of
  audio. This excludes loading and file I/O; it is not an iPhone benchmark.

## Standalone package checks (2026-10-25)

```sh
TEST_RUNNER_REUSE_STREAMING_CHECKPOINT="$PWD/Resources/realtime-reuse.safetensors" \
xcodebuild test -scheme ReuseApple-Package -configuration Release \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath BuildPackage \
  -clonedSourcePackagesDirPath SourcePackages ENABLE_TESTABILITY=YES CODE_SIGNING_ALLOWED=NO
bash Examples/StreamingApp/Checks/run_conversion_check.sh
bash Examples/Checks/run_report_check.sh
```

- All **16 package tests passed**, including checkpoint-backed GPU parity/reset
  tests. Those GPU tests skip when the checkpoint environment variable is absent.
- Both unsigned iOS example builds passed. The offline app bundle excludes the
  streaming checkpoint and its separate license.
- Offline GPU smoke, all four boundary checks, all six resampler cases and JSON
  report regression checks passed.
- Streaming CLI argument errors, overwrite/symlink protection, explicit `--force`,
  input preservation and JSON output checks passed.
- Streaming package output was **bit-identical** to the Mac prototype: 64,000
  finite samples at 16 kHz; both had STOI **0.669785048** on the paired diagnostic.

See [sanitized package evidence](benchmarks/package-validation.json).
These checks validate packaging and numerical preservation; they do not add a
new physical-iPhone timing result beyond the [prototype benchmark](streaming-validation.md).
