# Eight-layer native Vulkan backend

This experimental backend executes the complete fixed eight-layer streaming
model in FP32 Vulkan compute shaders. ONNX Runtime is used only by the offline
converter and test oracle. It is not invoked by the native GPU runtime.

**Latest physical comparison:** the previous v2 bundle averaged **17.69 ms** and
the v3 candidate **18.55 ms** in the same short phone test. Prefer v2 for now;
v3 was 4.87% slower. Both had p95 below 20 ms, but neither kept every hop below
20 ms. See the v3 comparison below for the protocol, quality checks and limits.

**Earlier v2 physical-phone result (October 26, 2026):** the optimized 530-dispatch
backend passed on a Galaxy S25 Ultra running Android 15 with an Adreno 830 GPU.
One foreground process measured the frozen tiled baseline and the optimized
bundle with 32 warmup hops and 300 timed hops per case:

- Frozen tiled Vulkan: **42.07 ms mean**, 41.74 ms median, 44.13 ms p95.
- Optimized tiled Vulkan: **22.48 ms mean**, 22.63 ms median, **24.65 ms p95**.
- Eight-thread CPU controls: **38.73 / 29.61 ms mean**, before/after.

The optimized GPU path reduced measured latency by **46.56%**, or **1.87×**
throughput relative to the tiled baseline in that execution. It **did not reach
the under-20-ms target**: 283 of 300 optimized hops exceeded 20 ms; the fastest
hop was 19.19 ms. The mean real-time factor was 1.124 for the 20-ms hop, so this
measurement does not establish sustained real-time processing. CPU controls
drifted by −23.56%, and cases ran in a fixed order without a counterbalanced
repeat. Treat the speedup as this recorded comparison, not a universal or
thermally controlled performance guarantee. The CPU default remains unchanged;
Vulkan stays explicitly experimental. Host/emulator timings are not phone
performance claims.

The optimized bundle passed unchanged FP32 waveform tolerances against a pinned
published-model CPU output: maximum absolute difference **1.58e-6** (limit
5e-6), relative RMS difference **3.33e-6** (limit 1e-5), STOI delta **2.37e-9**
and SI-SDR delta **4.40e-6 dB**. Its exported diagnostic waveform was bitwise
identical to the previous GPU baseline output on this sample. This does not
claim general CPU/GPU bitwise identity or broad perceptual validation: the
quality diagnostic is one fixed four-second synthetic noisy recording, without
alignment, gain adjustment or cropping. Reset and chunk replay were bitwise
stable within the optimized backend. Raw timings, identities, exported audio
and independent checks are in
`Validation/AndroidStreamingOptimization-vulkan-v2/physical-verification.json`;
Firebase matrix `matrix-3qiuchpunqkm4` and its JUnit result report success.
The checked-in result summary is
`android/benchmarks/streaming/results/firebase-vulkan-v2-galaxy-s25-ultra-2026-10-26.json`.

The original physical comparison remains preserved in
`Validation/AndroidStreamingOptimization-vulkan/physical-verification.json`
(matrix `matrix-20n6fdzg1cbf0`): direct Vulkan averaged 51.31 ms, tiled Vulkan
40.40 ms, and CPU controls 28.65 / 28.68 ms. Those are separate-run results;
the v2 comparison above remeasured the immutable baseline on the same
phone as the new candidate. The baseline package remains at `generated/final`;
the measured optimized package is `generated/candidate-v2`.

The converter accepts one SHA-pinned source graph. It folds static shape
calculations and constant transposes, aliases reshape-only views, preserves all
31 recurrent state tensors, and compiles the graph to a fixed dispatch plan.
Forty-eight SiLU pairs fuse sigmoid and multiplication into one dispatch while
retaining the same FP32 intermediate rounding boundaries. Bias/residual adds
join their matrix multiplications, and channel normalization operates directly
in convolution layout. The optional `--layout-fusion` plan also cancels inverse
transposes and fuses frequency depthwise convolution, its causal crop and SiLU;
delta projection also includes its Softplus. This reduces the original 798
dispatches to 530 without changing the mathematical topology. No layer, model
weight, hop, recurrent component, or output sample is dropped.

The additional opt-in `--flatten-matmul-batches` conversion treats each of 40
contiguous temporal projections `[161, 1, K]` as `[161, K]` with shared weights.
It only changes matrix rows, batch count and batch stride; every dot product
retains its original ordered input/weight addresses and bias/residual indexing.
This removes padded-row work, reducing tiled matmul workgroups from 43,744 to
7,744 while preserving all 530 dispatches, weights and the 104,001,280-byte
arena. The `adreno64` shader variant also executes each 64-channel normalization
row in one 64-lane subgroup, preserving the binary-tree sum order and avoiding
idle lanes/shared-memory barriers. Other admitted subgroup sizes use a portable
fallback. Neither change reduces precision or bypasses finite/state checks.

A separate 12-frame physical baseline profile identified those 40 temporal
matmuls as **17.138 ms of 33.495 ms whole-GPU time** (51.17%). These are
instrumented GPU intervals, not unprofiled end-to-end timings. The profile
validated operation identities against the exact binary plans and confirmed
profiled/unprofiled output equality; see
`Validation/AndroidStreamingOptimization-vulkan-v2/profile-analysis.json`.
The final comparison measured both variants with profiling disabled. It tests
the combined flattening and normalization changes, not their individual phone
speed contributions.

## V3 physical comparison and implementation

The v3 candidate at `generated/candidate-v3` passed its physical Galaxy S25
Ultra / Adreno 830 test on October 26, 2026. One foreground process compared
the previous v2 bundle and v3 in **ABBA order**, with 16 warmup hops and 160
timed hops per case. Pooling the two cases for each bundle gives 320 timed
public streaming calls per bundle:

- Previous v2: **17.690 ms mean**, **19.592 ms p95**, 20.757 ms maximum;
  **4 of 320** hops exceeded 20 ms.
- V3 candidate: **18.553 ms mean**, **19.592 ms p95**, 21.398 ms maximum;
  **4 of 320** hops exceeded 20 ms.

V3 was **4.87% slower** on mean latency, despite reducing dispatch count.
**Prefer the previous v2 bundle for now.** The v3 cache fusion and specialized
convolution remain experimental opt-ins. Both bundles met the under-20-ms
mean and p95 target in this short test; neither met a strict every-hop deadline.
These synchronous public-call timings include DSP, GPU completion and output
readback, but exclude audio-device and network latency. They do not establish
sustained thermal behavior or full application latency.

This phone ran firmware **AYD9**, while the earlier 22.48-ms v2 result above
used **AYB3** and 300 timed hops per case rather than 160. Do not attribute the
difference between those separate runs to v3: the unchanged v2 bundle was
faster than v3 in the new paired comparison. Within this ABBA run, v2's mean
changed by −2.75% between its first and last case; v3's repeat changed by −0.32%.

Both bundles passed the unchanged FP32 diagnostic gates: maximum absolute
waveform error **1.58e-6**, relative RMS error **3.33e-6**, STOI delta
**2.37e-9**, and SI-SDR delta **4.40e-6 dB** against the pinned published-model
CPU output. Reset replay, partial tails and cancellation checks passed. This
remains one four-second synthetic noisy-speech diagnostic, not broad perceptual
validation. Raw cases, identities and independent checks are preserved in
`Validation/AndroidStreamingOptimization-vulkan-v3/phone-verification.json`;
the result summary is
`android/benchmarks/streaming/results/firebase-vulkan-v3-galaxy-s25-ultra-2026-10-26.json`.

A separate four-hop candidate profile averaged **18.036 ms whole-GPU time**.
Its 13 specialized dense convolutions accounted for **5.957 ms (33.03%)**,
and 112 tiled matmuls for **5.141 ms (28.50%)**. These are instrumented GPU
intervals, not unprofiled public-call latency. The profile identifies the largest
remaining categories but cannot explain the v3 regression: there is no matched
v2 profile or phone ablation separating cache fusion from K72 convolution.
The larger K72 shared-memory allocation could affect occupancy, but that cause
was not measured. No further physical ablation was run within the testing cap.

The opt-in `--fuse-concat-cache` conversion replaces each of 41 two-piece
concatenations with a single exact-copy dispatch. For 23 of those operations,
the same loaded value also writes the retained cache suffix, eliminating the
following cache-copy dispatch. The original state input and its paired
next-state output remain disjoint; cache cropping retains exactly the original
state count. No floating-point arithmetic occurs in this fusion, including on
signed zero. Together these changes reduce **530 dispatches to 466**, retaining
all **31 recurrent state tensors**, all **eight layers**, FP32 precision, and
unchanged selected model-weight bytes. The v3 arena is 104,000,512 bytes; the
small arena-size change is an allocation-layout change, not model compression.

`--specialize-conv3x3` marks exactly 13 eligible dense 3×3 convolutions with
explicit descriptor flag `p27=1`. With tiled convolution enabled, the runtime
validates their shape and selects the separate `conv_3x3_tiled` pipeline
(internal key 208). It uses a 72-element reduction tile, covering eight input
channels, with constant 3×3 indexing and fewer synchronization points. The
ordered `input-channel → kernel-row → kernel-column` FP32 FMA accumulation,
bias addition and output indexing remain unchanged. The workgroup is still
16×8 threads, producing a tile of 16 output channels by 32 spatial positions.

The dedicated pipeline uses **13,824 bytes of shared memory**. The ordinary
`conv_tiled` fallback stays byte-identical to v2 and uses **3,072 bytes**, so
other convolution shapes do not inherit the larger allocation. Selection
requires the validated descriptor flag, not the presence of an extra shader
file. Existing v2 descriptors have `p27=0` and retain their original pipeline
mapping under the new runtime. Direct convolution, recurrence, model weights,
finite checks and commit semantics are unchanged.

Local evidence includes 19 exporter tests, 46 actual-GPU kernel cases and the
nonfinite-output rejection case. The four specialized convolution cases check
that `conv_3x3_tiled` actually executed, rather than silently passing through the
fallback. Cache cases verify bitwise copies including negative zero. All 16
selected SPIR-V files reproduce byte-identically and pass the FP32-only audit.
An eight-frame layout-fusion comparison matched v2 spectra and all 31 states
bitwise. The final 64-frame graph check passed the FP32 reference tolerances
for spectra and all 31 states. These checks establish local numerical behavior, not
phone latency or broad Vulkan-device portability. The subgroup-ordering
limitation described below still applies.

Selected shader artifacts and receipts are under
`generated/conv-v3/shaders72-separated-repro` and
`generated/conv-v3/selection-receipt.json`; kernel results are in
`generated/conv-v3/final-separated-kernel-tests/summary.json`. The specialized
plan is `generated/layout-v3/plan-specialized`. Keep the original baseline,
measured v2 package, and their validation history intact.

## Status and Android integration

These experimental sources and benchmark tools are available in this repository.
They are not included in the published Android `0.1.2` artifacts; this source
update does not create a Maven release. The v1, v2 and v3 physical experiments
above are complete. The latest
short test measured mean and p95 below 20 ms for both v2 and v3, with deadline
outliers in both. V2 is the preferred experimental GPU bundle on this evidence;
the CPU default remains unchanged.

The inference factories are `ExperimentalVulkanStreaming.create` and
`ExperimentalVulkanStreaming.createTiled`. Both use the same eight-layer FP32
weights and the exact plan from the supplied pinned bundle: 530 operations for
the measured v2 package or 466 for the v3 candidate. The experimental SDK pins
distinguish the candidate, measured v2 package and immutable baseline for
explicit comparisons. `createTiled` selects the separately validated tiled
matrix and eligible dense-convolution pipelines. Which is faster depends
on the phone's GPU and driver. Both factories preserve the existing
`StreamingEnhancer` push/finish/reset API; Apple APIs and CPU factories are
unchanged.

```kotlin
import io.riftco.reusestreaming.ExperimentalVulkanStreaming
import io.riftco.reusestreaming.StreamingEnhancerConfiguration
import java.io.File

// Extract the exact packaged model bundle into application-owned storage first.
val bundle = File(context.filesDir, "reuse-vulkan").absoluteFile
val library = File(context.applicationInfo.nativeLibraryDir,
    "libreuse_streaming_vulkan.so").absoluteFile
val configuration = StreamingEnhancerConfiguration(layers = 8, intraOpThreads = 1)

ExperimentalVulkanStreaming.createTiled(bundle, library, configuration).use { enhancer ->
    enhancer.warmUp()
    val output = enhancer.push(mono16kFloatSamples)
    val tail = enhancer.finish()
    // Consume output and tail in order. No input samples are dropped.
    println(enhancer.backendDiagnosticsJson)
}
```

Use `create` in the same call site to compare the direct kernels. CPU thread
count does not control Vulkan workgroups. The factory verifies the exact
manifest, all model/shader hashes and the native-library hash before loading.
`manifest.json`, `plan.bin`, `weights.bin` and the pinned `shaders/*.spv` files
must remain together. Package the ARM64 `.so` as an extracted JNI library and
preserve its bytes: APK stripping or replacing a shader invalidates the pins.
The benchmark's JNI packaging configuration demonstrates this setup. The
factory does not download model files or silently substitute another backend.

The network is implemented in C++ and Vulkan compute; ONNX Runtime is not linked
by the native library or used for GPU inference. The shared Kotlin SDK retains
its existing ONNX Runtime dependency for the separately selectable CPU backend.
Framing, STFT and overlap-add remain on the shared CPU FP32 DSP path. Benchmark
timing must include that DSP work, submission, GPU completion and output
readback, rather than measuring shader submission alone.

## Hardware and state contract

The Android build targets `arm64-v8a`, Android API 26 or newer, and requires a
Vulkan 1.1 hardware GPU. The device must support compute-stage subgroup basic,
shuffle and relative-shuffle operations, with subgroup width at least 16 and
divisible by 16. It also needs 128-byte push constants, workgroups of at least
256 invocations, and storage-buffer capacity for the checked model arena
(104,001,280 bytes for v2; 104,000,512 bytes for v3). The engine checks actual limits before
execution and rejects unsupported devices. CPU/software Vulkan devices are
rejected by Android JNI. The development CLI's explicit `--allow-software`
option is not a phone-performance mode.

There is a remaining portability limitation: the current normalization and
recurrent subgroup shaders assume local-invocation ordering corresponds to
subgroup-lane ordering. Vulkan does not generally guarantee that mapping, and
the existing device-limit checks do not establish it. The tested Adreno 830
passed the numerical/reference gates, but broader device support needs
subgroup-lane-safe indexing or stricter admission before production use. This
is a limitation of the measured artifacts, not a subsequent shader change.

Weights, intermediates and all 31 recurrent states remain on the GPU. Two
prerecorded command buffers address disjoint state banks. A frame is
provisional until GPU finite checks over its spectra and every next-state
tensor succeed, the fence completes, and the Kotlin cancellation check permits
commit. Commit switches the logical bank without copying the recurrent state.
Discarding a provisional frame leaves committed state unchanged; reset clears
both banks. Normal inference reads back 642 FP32 spectrum values plus one
status word. Full state readback is diagnostic only.

All shader arithmetic remains FP32; no FP16, quantization or layer reduction is
used. Explicit-FMA kernels and GPU reductions can produce small rounding
differences from the CPU reference. Cross-backend bitwise identity is not
claimed. Reset/replay within the same backend is tested bitwise, while waveform
and state agreement use the recorded numerical and speech-quality gates.

## Reproduce the optimized 530-operation FP32 package

Run from the repository root using the pinned development Python environment
with NumPy, ONNX, ONNX Runtime 1.23.2 and PyTorch. The converter requires the
already exported, SHA-pinned eight-layer source at
`tools/android-streaming-native/v2/softplus/generated/reuse-streaming-l8-native-stable-v2.onnx`;
it refuses an arbitrary replacement graph. Preserve the existing generated
model and parity fixtures when reproducing this experiment.

The build also needs shaderc (`glslc`), SPIR-V Tools (`spirv-val` and
`spirv-dis`), Android NDK **27.2.12479018**, CMake 3.22 or newer, and JDK 21.
Host GPU validation on macOS uses the Vulkan loader and MoltenVK. The native
build script records exact toolchain flags, source hashes, exported JNI symbols,
shared-library dependencies and artifact hashes in
`generated/native-build-receipt.json`.

Use a separate reproduction directory so the reviewed frozen package is not
replaced during development:

```sh
python tools/android-streaming-vulkan/export_plan.py \
  --layout-fusion --flatten-matmul-batches \
  --output tools/android-streaming-vulkan/generated/reproduction
python tools/android-streaming-vulkan/test_export_plan.py
python tools/android-streaming-vulkan/reference_plan.py \
  --plan tools/android-streaming-vulkan/generated/reproduction --frames 8
python tools/android-streaming-vulkan/build_shaders.py --math-variant fma \
  --gpu-variant adreno64 \
  --output tools/android-streaming-vulkan/generated/reproduction/shaders
python tools/android-streaming-vulkan/build_native.py

tools/android-streaming-vulkan/generated/build-host/reuse-vulkan-cli validate-plan \
  tools/android-streaming-vulkan/generated/reproduction/plan.bin
python tools/android-streaming-vulkan/test_gpu_kernels.py \
  --shaders tools/android-streaming-vulkan/generated/reproduction/shaders \
  --tiled-conv --tiled-matmul
python tools/android-streaming-vulkan/validate_gpu.py \
  --directory tools/android-streaming-vulkan/generated/reproduction \
  --shaders tools/android-streaming-vulkan/generated/reproduction/shaders \
  --output tools/android-streaming-vulkan/generated/reproduction/direct-validation
python tools/android-streaming-vulkan/validate_gpu.py \
  --directory tools/android-streaming-vulkan/generated/reproduction \
  --shaders tools/android-streaming-vulkan/generated/reproduction/shaders \
  --output tools/android-streaming-vulkan/generated/reproduction/tiled-validation \
  --tiled-conv --tiled-matmul
```

If the host Vulkan loader does not discover MoltenVK automatically, set
`VK_ICD_FILENAMES` to the installed `MoltenVK_icd.json` before running GPU tests.
For the Homebrew installation used here, that file is
`/opt/homebrew/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json`.

After these graph/kernel checks, package the exact selected artifacts and update
the experimental SDK pins for the subsequent SDK tests. This command only
assembles local benchmark assets; it does not publish a release:

```sh
python tools/android-streaming-vulkan/package_bundle.py \
  --generated tools/android-streaming-vulkan/generated/reproduction \
  --host-library tools/android-streaming-vulkan/generated/build-host/libreuse_streaming_vulkan.dylib \
  --android-library tools/android-streaming-vulkan/generated/build-arm64-v8a/libreuse_streaming_vulkan.so \
  --update-sdk-pins
```

The package contains the 530-operation layout-fused graph with 40 flattened
matmuls and the explicit-FMA `adreno64` shaders. To reproduce the historical
baseline instead, omit `--flatten-matmul-batches` and use
`--gpu-variant baseline`; write it into a different directory. Do not overwrite
`generated/final` or `generated/candidate-v2`. Packaging verifies source
identities, unchanged selected weight bytes, all shader hashes and FP32-only
SPIR-V audits. Keep `export.json`,
`shader-receipt.json`, `native-build-receipt.json` and `package-receipt.json`
alongside the validation results. Updating pins alone does not validate output.

## Reproduce the v3 466-operation candidate

Use the same pinned source, environment and toolchain described above. The
following flags select the cache/layout fusion and the separate K72 convolution
pipeline. Use a new output directory; never overwrite `generated/final`,
`generated/candidate-v2` or the reviewed `generated/candidate-v3` artifacts.

```sh
python tools/android-streaming-vulkan/export_plan.py \
  --layout-fusion --flatten-matmul-batches \
  --fuse-concat-cache --specialize-conv3x3 \
  --output tools/android-streaming-vulkan/generated/reproduction-v3
python tools/android-streaming-vulkan/test_export_plan.py
python tools/android-streaming-vulkan/reference_plan.py \
  --plan tools/android-streaming-vulkan/generated/reproduction-v3 --frames 8
python tools/android-streaming-vulkan/build_shaders.py \
  --math-variant fma --gpu-variant adreno64-conv72-separated \
  --output tools/android-streaming-vulkan/generated/reproduction-v3/shaders
python tools/android-streaming-vulkan/build_native.py \
  --output tools/android-streaming-vulkan/generated/reproduction-v3

tools/android-streaming-vulkan/generated/reproduction-v3/build-host/reuse-vulkan-cli validate-plan \
  tools/android-streaming-vulkan/generated/reproduction-v3/plan.bin
python tools/android-streaming-vulkan/test_gpu_kernels.py \
  --cli tools/android-streaming-vulkan/generated/reproduction-v3/build-host/reuse-vulkan-cli \
  --shaders tools/android-streaming-vulkan/generated/reproduction-v3/shaders \
  --output tools/android-streaming-vulkan/generated/reproduction-v3/kernel-tests \
  --tiled-conv --tiled-matmul --expect-dense-conv-pipeline
python tools/android-streaming-vulkan/validate_gpu.py \
  --directory tools/android-streaming-vulkan/generated/reproduction-v3 \
  --cli tools/android-streaming-vulkan/generated/reproduction-v3/build-host/reuse-vulkan-cli \
  --shaders tools/android-streaming-vulkan/generated/reproduction-v3/shaders \
  --output tools/android-streaming-vulkan/generated/reproduction-v3/tiled-validation \
  --tiled-conv --tiled-matmul
```

The separate-pipeline shader variant requires `--math-variant fma`. Its
`conv_tiled.spv` is the original FMA fallback; only marked descriptors select
`conv_3x3_tiled.spv`. Packaging checks the dedicated shader hash as part of the
bundle rather than admitting arbitrary additional shaders by file presence.
After graph/kernel validation, prepare local SDK assets with:

```sh
python tools/android-streaming-vulkan/package_bundle.py \
  --generated tools/android-streaming-vulkan/generated/reproduction-v3 \
  --host-library tools/android-streaming-vulkan/generated/reproduction-v3/build-host/libreuse_streaming_vulkan.dylib \
  --android-library tools/android-streaming-vulkan/generated/reproduction-v3/build-arm64-v8a/libreuse_streaming_vulkan.so \
  --update-sdk-pins
```

Then run the public SDK tests below using the `reproduction-v3` bundle and its
host library. Packaging and local parity checks do not publish a release,
submit a Firebase matrix, or establish physical-phone performance.

## SDK tests and physical benchmark

`VulkanStreamingIntegrationTest` runs the actual public streaming path against
the published FP32 CPU graph. Supply absolute paths to the local fixtures; the
GPU tests intentionally skip when these explicit opt-in variables are absent:

```sh
export REUSE_VULKAN_DIRECTORY="$PWD/tools/android-streaming-vulkan/generated/reproduction"
export REUSE_VULKAN_LIBRARY="$PWD/tools/android-streaming-vulkan/generated/build-host/libreuse_streaming_vulkan.dylib"
export REUSE_STREAMING_MODEL_FILE="$PWD/tools/streaming-export/generated/models/reuse-streaming-l8-fp32.onnx"
export REUSE_STREAMING_MODEL_MANIFEST="$PWD/android/reuse-streaming/model/manifests/reuse-streaming-l8-fp32.manifest.json"
export REUSE_STREAMING_PARITY_DIR="$PWD/tools/streaming-export/generated/parity/l8"
export REUSE_VULKAN_QUALITY_INPUT="$PWD/Validation/AndroidStreamingOptimization-0.1.2/quality/noisy-speech.f32"
export REUSE_VULKAN_QUALITY_OUTPUT="$PWD/tools/android-streaming-vulkan/generated/reproduction/quality"

./android/gradlew -p android :reuse-streaming:testDebugUnitTest \
  --tests io.riftco.reusestreaming.VulkanStreamingIntegrationTest
```

Verify the XML reports show execution with **zero skipped tests**. The suite
covers seven waveform lengths, chunk boundaries, EOF tails, all carried states,
reset/replay, cancellation, direct/tiled parity and the four-second synthetic
speech diagnostic. The diagnostic is a numerical/audio-quality regression gate,
not a broad perceptual-quality study. The native CLI additionally supports
`--transaction-test` to check discarded GPU frames, rejected nonfinite input,
exactly-once commit and state-bank rollback.

The original benchmark instrumentation is
`android/benchmarks/streaming/app/src/vulkanAndroidTest/.../VulkanBenchmarkTest.kt`,
with historical artifact validation in `verify_phone_result.py`.
`VulkanOptimizedBenchmarkTest.kt` compares four ordered cases on one phone:
CPU before, frozen tiled GPU, optimized tiled GPU, CPU after. It records raw
synchronous public-push timings, exact workload and bundle identities, actual
Vulkan GPU/driver/pipeline information, output lengths and quality waveforms.
Its quality reference is the SHA-pinned published CPU output from the prior
physical run using identical input, rather than a newly generated CPU waveform.
`verify_optimized_result.py` independently validates both bundles, disabled
profiling, 300 physical timed hops, the raw statistics, golden CPU identity and
waveform quality. The verifier's negative tests use marked temporary structural
fixtures; those fixtures are rejected as production benchmark evidence.

`VulkanFinalBudgetBenchmarkTest.kt` records the newer v2/v3 ABBA comparison:
two 160-hop cases per bundle, 16 warmup hops per case, with profiling disabled
during latency measurements. `verify_final_budget_result.py` independently
checks artifact identities, all raw timings, pooled statistics and exported
quality waveforms. Its optional four-hop v3 profile is recorded separately and
verifies profiled/unprofiled output equality; it is not a paired kernel ablation.

For diagnosis, `ExperimentalVulkanStreaming.createProfiled(bundle, library,
configuration, tiled = true)` enables per-dispatch timestamps. Read
`enhancer.backendProfileJson` after a push for the last submitted frame;
`backendDiagnosticsJson` reports `perDispatchProfiling`. The development CLI
also accepts `--profile`. `VulkanProfileTest.kt` captures a short baseline
profile and verifies its output against the unprofiled backend. Timestamp
queries perturb scheduling, and per-dispatch intervals differ from full public
push latency. Never mix profiled and unprofiled cases when claiming a speedup,
or treat a negative measured profiling overhead in a short sequential capture
as proof that instrumentation is free. Use `create`/`createTiled`, with
`perDispatchProfiling = false`, for latency comparisons.

Host and emulator tests are free local checks. A Firebase physical run is a
separate paid action governed by the cumulative testing receipt; the build and
validation commands above never submit a cloud test. Do not infer phone results
from host GPU timings, or rerun paid matrices without accounting for prior
spending and the existing test cap.

## Plan format, version 1

All integer fields are little-endian uint32; arena offsets count FP32 elements.
`plan.bin` contains:

1. Eight ASCII bytes `RVKPLAN1`.
2. Six uint32 fields: version, arena float count, dispatch count, input count,
   output count, state-pair count.
3. Input records followed by output records, each `(offset, element_count)`.
4. State records `(input_offset, output_offset, element_count)`.
5. Dispatch records of 33 uint32: 32 push-constant words and an address mask.
   Bit `i` in the last word marks word `i` as an arena address. The runtime
   remaps those addresses for the alternate recurrent-state bank.

`weights.bin` is the complete initial arena: immutable FP32 constants, zero
recurrent state and scratch space. `export.json` records full names, dimensions,
state mappings, source initializer hashes, artifact hashes and dispatch audit.
Its `files` entries can be incorporated into the final package manifest along
with compiled shader hashes. The fixed plan currently requires about 99 MiB,
below Vulkan's guaranteed 128 MiB storage-buffer range. Inputs and next-state
outputs have dedicated, disjoint allocations; only intermediates use liveness
reuse. The native engine validates actual device limits before execution.

Kernel push-constant fields are defined by the shader ABI documentation.
Persistent state remains on the GPU. A frame does not become the next frame's
state until the finite-value check succeeds and its caller commits it. Normal
operation transfers only the two 321-bin input/output spectra and a status
word; full state readback exists only for validation.

The model remains the NVIDIA Real-time RE-USE eight-layer checkpoint, with its
original NSCLv1 noncommercial research/education license. This experimental
backend does not change the model license. MLAS-derived Exp code retains its
upstream MIT notice. Apple APIs and existing Android CPU factories remain
unchanged.
