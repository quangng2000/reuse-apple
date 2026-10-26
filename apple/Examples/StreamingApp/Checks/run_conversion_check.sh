#!/bin/bash
set -euo pipefail
checks="$(cd "$(dirname "$0")" && pwd)"
temporary="$(mktemp -d "${TMPDIR:-/tmp}/reuse-conversion.XXXXXX")"
trap 'rm -rf "$temporary"' EXIT
# Exercise the actual conversion implementation with a local error stub, without
# loading the ML model or asking for microphone permission.
sed '/^import ReuseAudio$/d' "$checks/../Audio/MicrophoneConversion.swift" > "$temporary/MicrophoneConversion.swift"
xcrun swiftc -O "$temporary/MicrophoneConversion.swift" "$checks/ConversionDrainCheck.swift" \
  -o "$temporary/conversion-check"
"$temporary/conversion-check"
