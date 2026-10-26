#!/usr/bin/env bash
set -euo pipefail
source_directory="$1"
output_directory="$2"
jdk_directory="$3"
mkdir -p "$output_directory"
case "$(uname -s)" in
  Darwin) jni_platform=darwin; library_name=libreuseaudio_dsp.dylib ;;
  Linux) jni_platform=linux; library_name=libreuseaudio_dsp.so ;;
  *) echo "Host DSP tests require a macOS/Linux C++17 compiler and JDK." >&2; exit 1 ;;
esac
"${CXX:-c++}" -std=c++17 -O2 -fno-fast-math -ffp-contract=on -DPOCKETFFT_NO_MULTITHREADING=1 -fPIC -shared \
  -I"$jdk_directory/include" -I"$jdk_directory/include/$jni_platform" \
  "$source_directory/reuse_audio_dsp.cpp" -o "$output_directory/$library_name"
