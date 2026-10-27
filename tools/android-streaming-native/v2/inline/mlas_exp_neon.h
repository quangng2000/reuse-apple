// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License (see MLAS-LICENSE).
// Explicit vfmaq preserves the fused arithmetic of the shipped ARM64 ORT build
// even though state-update compilation disables implicit FP contraction.
// ARM NEON specialization of ONNX Runtime v1.23.2 compute.cpp's
// MlasComputeExpVector, with the same constants and mlasi.h operation order.
#pragma once
#if !defined(__aarch64__) && !defined(_M_ARM64)
#error "This explicit inline V2 experiment requires ARM64 NEON"
#endif
#include <arm_neon.h>

inline float32x4_t ReuseMlasExpVector(float32x4_t value) {
  value = vmaxq_f32(vdupq_n_f32(-103.9720840454f), value);
  value = vminq_f32(vdupq_n_f32(88.7762626647950f), value);
  const auto bias = vdupq_n_f32(12582912.f);
  const auto biased = vfmaq_f32(bias, value, vdupq_n_f32(1.44269504088896341f));
  const auto m = vsubq_f32(biased, bias);
  value = vfmaq_f32(value, m, vdupq_n_f32(-6.93145752e-1f));
  value = vfmaq_f32(value, m, vdupq_n_f32(-1.42860677e-6f));

  const auto minimum = vdupq_n_s32(static_cast<int32_t>(0xC1000000));
  const auto maximum = vdupq_n_s32(static_cast<int32_t>(0x3F800000));
  auto overflow = vshlq_n_s32(vreinterpretq_s32_f32(biased), 23);
  auto normal = vmaxq_s32(vminq_s32(overflow, maximum), minimum);
  overflow = vaddq_s32(vsubq_s32(overflow, normal), maximum);
  normal = vaddq_s32(normal, maximum);

  auto p = vdupq_n_f32(0x1.694000p-10f);
  p = vfmaq_f32(vdupq_n_f32(0x1.125edcp-7f), p, value);
  p = vfmaq_f32(vdupq_n_f32(0x1.555b5ap-5f), p, value);
  p = vfmaq_f32(vdupq_n_f32(0x1.555450p-3f), p, value);
  p = vfmaq_f32(vdupq_n_f32(0x1.fffff6p-2f), p, value);
  p = vfmaq_f32(vdupq_n_f32(0x1.000000p+0f), p, value);
  value = vmulq_f32(value, vreinterpretq_f32_s32(overflow));
  p = vfmaq_f32(vreinterpretq_f32_s32(overflow), p, value);
  return vmulq_f32(p, vreinterpretq_f32_s32(normal));
}

inline void ReuseMlasExp16(float delta, const float* coefficients, float* decay) {
  for (int index = 0; index < 16; index += 4) {
    // Keep the original standalone Mul rounding before Exp; this is not a
    // fast-math approximation or a contraction of the state Mul/Add update.
    const auto product = vmulq_f32(vdupq_n_f32(delta), vld1q_f32(coefficients + index));
    vst1q_f32(decay + index, ReuseMlasExpVector(product));
  }
}
