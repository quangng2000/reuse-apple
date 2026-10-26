#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ "$(uname -m)" != arm64 ]]; then
  echo 'RE-USE requires an Apple Silicon Mac; launch a native arm64 shell.' >&2
  exit 1
fi
xcode_version="$(xcodebuild -version 2>/dev/null | awk '/^Xcode / {print $2}' || true)"
IFS=. read -r xcode_major xcode_minor <<< "$xcode_version"
if [[ ! "${xcode_major:-}" =~ ^[0-9]+$ ]] ||
   (( xcode_major < 16 || (xcode_major == 16 && ${xcode_minor:-0} < 4) )); then
  echo 'Select full Xcode 16.4 or newer with xcode-select (or DEVELOPER_DIR).' >&2
  exit 1
fi
mkdir -p "$project_dir/Logs"
python3 "$project_dir/tools/model-export/prepare_model.py"
if ! (cd "$project_dir" && xcodebuild build -scheme ReuseProbe \
  -configuration Release -destination 'platform=macOS,arch=arm64' -derivedDataPath "$project_dir/Build" \
  -clonedSourcePackagesDirPath "$project_dir/SourcePackages" -jobs 4 \
  ARCHS=arm64 CODE_SIGNING_ALLOWED=NO > "$project_dir/Logs/build-mac.log" 2>&1); then
  tail -n 50 "$project_dir/Logs/build-mac.log" >&2
  exit 1
fi
export REUSE_MODEL_PATH="$project_dir/models/model.safetensors"
exec "$project_dir/Build/Build/Products/Release/ReuseProbe" "$@"
