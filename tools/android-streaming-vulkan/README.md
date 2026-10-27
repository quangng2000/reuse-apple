# Eight-layer native Vulkan backend

This experimental backend executes the complete fixed eight-layer streaming
model in FP32 Vulkan compute shaders. ONNX Runtime is used only by the offline
converter and test oracle. It is not invoked by the native GPU runtime.

**Current selection: v2 only.** `ExperimentalVulkanStreaming.create()` uses the
measured tiled v2 kernels. `createTiled()` is a compatibility alias for the same
path. The SDK accepts only the v2 bundle manifest SHA-256
`951adf47ebece2c27a5af4df22b91c30c8a442b8caa04bb01db102ceb0ac733a`;
the original v1 baseline and v3 candidate are rejected. The ordinary CPU
constructor and Apple APIs are unchanged.

**Historical physical comparison:** v2 averaged **17.69 ms** and v3 **18.55 ms**
in the same short Galaxy S25 Ultra test; v3 was 4.87% slower. Both had p95 below
20 ms, but each had 4/320 calls above 20 ms. Selecting v2 does not establish a
new benchmark result or a sustained real-time guarantee. The archived v3
comparison below records the protocol, quality checks and limits.

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

## Archived v3 physical comparison and implementation

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
**The current SDK selects v2 and rejects v3.** The v3 cache fusion and
specialized convolution remain archived code-generation experiments, not SDK
backend options. Both bundles met the under-20-ms
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

The archived `--fuse-concat-cache` conversion replaces each of 41 two-piece
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
The published Android `0.1.2` artifacts predate the Vulkan API. Android `0.1.3`
provides the v2-only Kotlin API; the model bundle and pinned native library remain
separate artifacts. The v1, v2 and v3 physical experiments
above are complete. The latest
short test measured mean and p95 below 20 ms for both v2 and v3, with deadline
outliers in both. V2 is the preferred experimental GPU bundle on this evidence;
the CPU default remains unchanged.

The Vulkan factories accept the same single eight-layer FP32 v2 bundle: 530
operations, 40 flattened matmuls, the measured `adreno64` FP32 shaders and
31 recurrent states. `create()` selects the tiled matrix and eligible
convolution pipelines by default; `createTiled()` is an equivalent compatibility
alias. There is no public unprofiled direct-kernel selection. Older bundle
manifests are rejected before loading. The separately pinned current native
library is unchanged by this selection update. Both factory names preserve the
existing `StreamingEnhancer` push/finish/reset API.

```kotlin
import io.riftco.reusestreaming.ExperimentalVulkanStreaming
import io.riftco.reusestreaming.StreamingEnhancerConfiguration
import java.io.File

// Extract the exact audited v2 bundle into application-owned storage first.
val bundle = File(context.filesDir, "reuse-vulkan-v2").absoluteFile
val library = File(context.applicationInfo.nativeLibraryDir,
    "libreuse_streaming_vulkan.so").absoluteFile
val configuration = StreamingEnhancerConfiguration(layers = 8, intraOpThreads = 1)

ExperimentalVulkanStreaming.create(bundle, library, configuration).use { enhancer ->
    enhancer.warmUp()
    val output = enhancer.push(mono16kFloatSamples)
    val tail = enhancer.finish()
    // Consume output and tail in order. No input samples are dropped.
    println(enhancer.backendDiagnosticsJson)
}
```

`createTiled` is an alias, not a different backend. CPU thread count does not
control Vulkan workgroups. The factory verifies the exact
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
(104,001,280 bytes for the accepted v2 bundle). The engine checks actual limits before
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

## Use the audited v2 artifacts

Use the original audited `generated/candidate-v2` bundle, or an unchanged copy
of it, together with the currently pinned native library. Generated bundles and
native binaries are local research artifacts and are not included in the AAR or
this repository. The factory does not download them. Verify the
manifest before staging it into application-owned storage:

```sh
python3 - /absolute/path/to/candidate-v2/manifest.json <<'PYVERIFY'
import hashlib
import pathlib
import sys
expected = "951adf47ebece2c27a5af4df22b91c30c8a442b8caa04bb01db102ceb0ac733a"
assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest() == expected
PYVERIFY
```

The manifest binds the original source graph, plan, weights, shaders and export
receipts. `manifest.json`, `plan.bin`, `weights.bin` and all listed shader files
must remain byte-identical. Recompiling an equivalent graph or regenerating a
receipt does not automatically reproduce this manifest identity. Do not change
SDK pins to admit a different bundle merely because it has eight layers or
passes a single numerical test.

To stage the existing audited package for a local benchmark build, supply its
unchanged export/shader receipts and the exact current native libraries:

```sh
python tools/android-streaming-vulkan/package_bundle.py \
  --generated /absolute/path/to/candidate-v2 \
  --host-library /absolute/path/to/current-pinned/libreuse_streaming_vulkan.dylib \
  --android-library /absolute/path/to/current-pinned/libreuse_streaming_vulkan.so
```

Active staging and `--update-sdk-pins` admit only that measured v2 manifest and
the existing validated native-library hashes. The optional pin-update flag
does not admit a newly generated or historical alternative. Packaging stages
local files only; it does not download a bundle, run a paid test or publish a
release.

The converter and native CLI remain available for development: the v2 graph
uses `export_plan.py --layout-fusion --flatten-matmul-batches`, and its shaders
use `build_shaders.py --math-variant fma --gpu-variant adreno64`. Development
outputs are not automatically admitted by the SDK. NumPy, ONNX, ONNX Runtime
1.23.2, PyTorch, shaderc/SPIR-V Tools, NDK 27.2.12479018 and CMake 3.22 or newer
are the recorded toolchain. Host Vulkan checks on macOS additionally need the
Vulkan loader and MoltenVK. The generator still requires the pinned source at
`tools/android-streaming-native/v2/softplus/generated/reuse-streaming-l8-native-stable-v2.onnx`.
Keep generated outputs in a separate ignored directory and preserve the audited
v2 artifacts.

## Archived generation and comparison experiments

The original v1 and v3 generators, shader variants, validators and numerical
results are retained for research history. Their presence does not make those
bundles acceptable to the current SDK. In particular, `--fuse-concat-cache` and
`--specialize-conv3x3` produce the archived v3 candidate, which the SDK rejects.
`package_bundle.py --package-only` can write a development manifest and receipt
without staging Android assets or modifying SDK pins. It cannot be combined
with `--update-sdk-pins`, and does not make that bundle admissible at runtime.

To reproduce the historical direct/tiled, v1/v2 or v2/v3 comparisons, use a
separate checkout of [commit ad4ceda](https://github.com/quangng2000/reuse-audio/tree/ad4ceda)
with its matching SDK, exact frozen bundles, native library and fixtures. That
commit's version of this README records the original generation and comparison
commands. The old instrumentation source is also retained under
`android/benchmarks/streaming/archive/vulkan/`, outside active instrumentation
source sets. Do not copy it into the current benchmark to bypass v2-only
admission. The recorded result JSONs remain historical evidence; this selection
change does not rerun or reinterpret their timings.

## SDK tests and physical benchmark

`VulkanStreamingIntegrationTest` runs the actual public streaming path against
the published FP32 CPU graph. Supply absolute paths to the local fixtures; the
GPU tests intentionally skip when these explicit opt-in variables are absent:

```sh
export REUSE_VULKAN_DIRECTORY="/absolute/path/to/candidate-v2"
export REUSE_VULKAN_LIBRARY="/absolute/path/to/current-pinned/libreuse_streaming_vulkan.dylib"
export REUSE_STREAMING_MODEL_FILE="$PWD/tools/streaming-export/generated/models/reuse-streaming-l8-fp32.onnx"
export REUSE_STREAMING_MODEL_MANIFEST="$PWD/android/reuse-streaming/model/manifests/reuse-streaming-l8-fp32.manifest.json"
export REUSE_STREAMING_PARITY_DIR="$PWD/tools/streaming-export/generated/parity/l8"
export REUSE_VULKAN_QUALITY_INPUT="$PWD/Validation/AndroidStreamingOptimization-0.1.2/quality/noisy-speech.f32"
export REUSE_VULKAN_QUALITY_OUTPUT="$PWD/tools/android-streaming-vulkan/generated/v2-validation/quality"

./android/gradlew -p android :reuse-streaming:testDebugUnitTest \
  --tests io.riftco.reusestreaming.VulkanStreamingIntegrationTest
```

Verify the XML reports show execution with **zero skipped tests**. The suite
covers seven waveform lengths, chunk boundaries, EOF tails, all carried states,
reset/replay, cancellation, factory-alias parity and the four-second synthetic
speech diagnostic. The diagnostic is a numerical/audio-quality regression gate,
not a broad perceptual-quality study. The native CLI additionally supports
`--transaction-test` to check discarded GPU frames, rejected nonfinite input,
exactly-once commit and state-bank rollback.

Historical instrumentation lives in
`android/benchmarks/streaming/archive/vulkan/` and is excluded from active test
source sets. `VulkanBenchmarkTest.kt` recorded direct/tiled v1 measurements;
`VulkanOptimizedBenchmarkTest.kt` recorded CPU/v1/v2/CPU comparisons;
`VulkanFinalBudgetBenchmarkTest.kt` recorded v2/v3 ABBA comparisons. Reproduce
those protocols with commit `ad4ceda`, not with the v2-only current SDK.

Their independent validators (`verify_phone_result.py`,
`verify_optimized_result.py` and `verify_final_budget_result.py`) remain useful
for checking archived result identities, raw timings and quality waveforms.
Marked structural fixtures in verifier tests are not physical benchmark
evidence. No new physical run was performed for the v2-only selection change.

For diagnosis, `ExperimentalVulkanStreaming.createProfiled(bundle, library,
configuration, tiled = true)` enables per-dispatch timestamps. Read
`enhancer.backendProfileJson` after a push for the last submitted frame;
`backendDiagnosticsJson` reports `perDispatchProfiling`. The development CLI
also accepts `--profile`. The active `VulkanProfileTest.kt` is a v2-only
diagnostic and checks profiled output against the unprofiled backend. It is not
one of the archived multi-bundle comparison tests. Current profiling accepts
only the v2 bundle; use `tiled = true` to diagnose the selected tiled path. Timestamp
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
