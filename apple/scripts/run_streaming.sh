#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ "$(uname -m)" != arm64 ]]; then
  echo 'Streaming RE-USE requires native Apple Silicon.' >&2
  exit 1
fi
needs_default_model=true
for argument in "$@"; do
  if [[ "$argument" == --model || "$argument" == --help || "$argument" == -h ]]; then
    needs_default_model=false
  fi
done
if $needs_default_model; then
  python3 "$project_dir/apple/scripts/prepare_streaming_model.py"
  set -- "$@" --model "$project_dir/models/realtime-reuse.safetensors"
fi
mkdir -p "$project_dir/Logs"
if ! (cd "$project_dir" && xcodebuild -scheme ReuseStreamingCLI -configuration Release \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath "$project_dir/BuildStreaming" \
  -clonedSourcePackagesDirPath "$project_dir/SourcePackages" CODE_SIGNING_ALLOWED=NO \
  build > "$project_dir/Logs/build-streaming.log" 2>&1); then
  tail -n 50 "$project_dir/Logs/build-streaming.log" >&2
  exit 1
fi
exec "$project_dir/BuildStreaming/Build/Products/Release/ReuseStreamingCLI" "$@"
