# Public streaming benchmark evidence

These reports preserve the recorded timings, numerical and audio-quality checks,
model identities, device specifications and artifact hashes for the Android
streaming experiments. They are public copies of locally retained test evidence.
Personal project/account identifiers, cloud-storage and console locations,
provider administration records, financial budgets and authorization text, and
local filesystem paths have been removed. The original receipts remain unchanged
outside the repository. Recorded execution durations are not invoices or billing
claims.

Each JSON has an adjacent `.json.sha256` containing its public file's SHA-256.
Existing fields such as `resultSha256`, source hashes and APK/model hashes still
identify the originally captured records or named binary artifacts. They do not
claim to hash a redacted nested object or the public report as a whole. Source
and model hashes provide provenance; they do not replace rerunning the numerical
checks or prove phone performance. Model binaries, recordings, APKs and private
provider receipts are not included here.

## Latest comparison: keep v2 as the experimental GPU choice

The final Galaxy S25 Ultra / Adreno 830 comparison used ABBA order, with 16
warmup hops and 160 measured public streaming calls per block. The two blocks
for each backend total 320 measurements:

- Previous Vulkan v2: **17.69 ms mean**, **19.59 ms p95**; 4/320 calls above 20 ms.
- Vulkan v3 candidate: **18.55 ms mean**, **19.59 ms p95**; 4/320 calls above 20 ms.

V3 was **4.87% slower** in that same-run comparison. Prefer the explicit v2
experimental GPU bundle; retain v3 for further investigation. Both means and
p95 values were below 20 ms, but neither version met an every-hop deadline.
The earlier 22.48-ms v2 measurement used different firmware and a different
protocol, so its difference from this result is not evidence of a v3 gain.

The two GPU outputs were bitwise identical on the four-second diagnostic clip.
Both had small FP32 rounding differences from the pinned CPU reference. This is
one synthetic noisy-speech quality diagnostic, not general bitwise equivalence
or broad perceptual validation. Timings include DSP, GPU completion and readback;
they exclude audio-device/network latency and do not establish sustained thermal
or battery behavior. The separate four-hop v3 GPU profile was instrumented and
is excluded from these latency statistics.

The reports distinguish physical Android measurements from host/emulator results.
The CPU default and Apple APIs are unchanged. This source push is **not a Maven
release**, does not publish model weights or native bundles, and does not change
the existing Android 0.1.2 artifacts. The experimental eight-layer streaming
checkpoint and derivative inference implementation retain their NSCLv1
noncommercial research/education restrictions. Broader Vulkan-device subgroup
portability requires further validation.

See [the Vulkan implementation notes](../../../../tools/android-streaming-vulkan/README.md)
for reproduction commands, numerical tolerances and backend limitations.
