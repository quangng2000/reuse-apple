# Native streaming Vulkan engine

This directory implements the experimental eight-layer network directly with C++ and Vulkan compute. ONNX Runtime is not linked or invoked. Streaming framing, STFT and overlap-add remain in the existing shared FP32 SDK DSP path; the network weights, intermediate tensors and all 31 recurrent state tensors stay in GPU memory.

The runtime accepts only version-one fixed-shape plans. It validates record lengths, operation IDs, address masks, tensor bounds, shapes, state mappings and disjoint state banks before GPU allocation. The public Kotlin factory additionally verifies the exact model bundle and native-library SHA-256 identities. A Vulkan 1.1 hardware GPU with the required subgroup operations and workgroup/storage limits is mandatory. Android never silently falls back to CPU inference.

Two reusable command buffers implement recurrent ping-pong banks. `frame` computes provisional output and performs GPU nonfinite checks over both output spectra and every next-state tensor. It waits for a GPU fence, reads back only 642 floats plus the finite-check flag, and leaves committed state unchanged. `commit` changes the logical bank without copying the state. A canceled or rejected frame is discarded by leaving the bank unchanged. `reset` zeros both recurrent banks. State readback is an explicit validation-only operation, outside normal inference timing.

`infoJson` reports the actual selected GPU, Vulkan and driver versions, subgroup capabilities, arena size, operation count and selected tiled pipelines. GPU timestamps are diagnostic; user-facing streaming benchmarks time the full public push call, including CPU DSP, submission, fence wait, finite checks and output transfer.

## Build and local validation

The Android build uses the installed NDK and links the Android Vulkan loader. Host validation uses the Vulkan loader and MoltenVK on the Mac GPU. Neither host measurements nor emulator measurements substitute for a physical Android phone benchmark.

```sh
cmake -S tools/android-streaming-vulkan/native \
  -B tools/android-streaming-vulkan/generated/build-host -DCMAKE_BUILD_TYPE=Release
cmake --build tools/android-streaming-vulkan/generated/build-host -j 4
python3 tools/android-streaming-vulkan/native/test_plan_validation.py

cmake -S tools/android-streaming-vulkan/native \
  -B tools/android-streaming-vulkan/generated/build-arm64-v8a \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DCMAKE_BUILD_TYPE=Release
cmake --build tools/android-streaming-vulkan/generated/build-arm64-v8a -j 4
```

The CLI input and output files contain little-endian FP32 frames, each with 321 magnitudes followed by 321 phases. Its output contains wall-clock and GPU model-call timings, not end-to-end waveform timings. `--states` writes the final committed state tensors for oracle comparison. `--reset-replay` reruns the input from zero state and requires bitwise-repeatable results. Optional `--tiled-matmul` and `--tiled-conv` select separately validated dense-kernel variants.

```sh
reuse-vulkan-cli PLAN.bin WEIGHTS.bin SHADER_DIRECTORY INPUT.f32 OUTPUT.f32 \
  --states STATE_DIRECTORY --reset-replay
reuse-vulkan-cli validate-plan PLAN.bin
```

The diagnostic CLI has an explicit `--allow-software` option for development only. The Android JNI entry points always disable software-device selection. FP16, quantization and model-layer reduction are not implemented in this backend.
