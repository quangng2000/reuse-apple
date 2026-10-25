#!/bin/bash
set -euo pipefail
checks="$(cd "$(dirname "$0")" && pwd)"
temporary="$(mktemp -d "${TMPDIR:-/tmp}/reuse-report.XXXXXX")"
trap 'rm -rf "$temporary"' EXIT
for source in "$checks/../StreamingCLI/StreamingOptions.swift" \
              "$checks/../StreamingCLI/StreamingReport.swift" \
              "$checks/../StreamingApp/Audio/RecordingReport.swift"; do
  sed '/^import ReuseAudio$/d' "$source" > "$temporary/$(basename "$source")"
done
xcrun swiftc -O "$temporary/StreamingOptions.swift" "$temporary/StreamingReport.swift" \
  "$temporary/RecordingReport.swift" "$checks/ReportSerializationCheck.swift" \
  -o "$temporary/report-check"
"$temporary/report-check"
