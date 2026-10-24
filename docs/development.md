# Build and check

The generated Xcode project is included; XcodeGen is only needed when changing
the project definition. Swift package revisions are locked in `Package.resolved`.

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
xcodebuild build -project ReusePocket.xcodeproj -scheme ReusePocket \
  -configuration Release -destination 'generic/platform=iOS' \
  -derivedDataPath BuildIOS -clonedSourcePackagesDirPath SourcePackages \
  CODE_SIGNING_ALLOWED=NO
```

An unsigned compile does not install the app. Device testing requires Xcode
signing with your own team and connected hardware. No signing credentials are
included in this repository.

After adding or moving Swift sources, regenerate the project:

```sh
xcodegen generate --spec project.yml
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
