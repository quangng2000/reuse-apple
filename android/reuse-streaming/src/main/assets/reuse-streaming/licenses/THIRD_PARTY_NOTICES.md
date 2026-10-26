# Android RE-USE streaming notices

The Real-time RE-USE port and model are adaptations of NVIDIA Real-time RE-USE,
revision `4c0c186272ca853cd18d832fc5c28777b62f32d1`, under the NVIDIA One-Way
Noncommercial License (NSCLv1). Copyright (c) 2025 NVIDIA CORPORATION.
All rights reserved. Noncommercial research and education only. See LICENSE.txt.
This is a separate optional research module, not the commercial 30-layer SDK.

The FP32 PocketFFT header is copied byte-for-byte from the pinned offline SDK
header, under the BSD 3-Clause license; see POCKETFFT_LICENSE.txt. The JNI
adapter is copied from that SDK with only its JNI symbol prefix changed and
its MIT license retained; see ADAPTER_LICENSE.txt. No neural model code from
the commercial SDK is included. The separate native library is reusestreaming_dsp.

ONNX Runtime 1.23.2 is a runtime dependency under the MIT License. Its license
and third-party notices are included as ONNXRUNTIME_LICENSE.txt and
ONNXRUNTIME_THIRD_PARTY_NOTICES.txt. The runtime license does not grant commercial
rights to the NSCLv1 model or its derivative port.
