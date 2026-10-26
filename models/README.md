# Shared model metadata

This directory contains the pinned offline checkpoint's provenance, upstream model
card and required license/notices. Both native backends use the same 30-layer FP32
model identity recorded in `ModelProvenance.json`.

Downloaded `.safetensors` weights stay out of Git. Fetch the offline checkpoint
from the repository root with `python3 tools/model-export/prepare_model.py`.
The Android ONNX export and its receipt are generated under
`tools/model-export/generated/`; those binaries are also excluded from Git.

The separate Apple streaming research model has its own provenance and license
under [`apple/Research`](../apple/Research). It is not part of the Android SDK.
