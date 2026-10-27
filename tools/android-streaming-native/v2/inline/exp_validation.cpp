#include "mlas_exp_neon.h"
#include <cstddef>
extern "C" __attribute__((visibility("default"))) void ReuseExpArrayForValidation(const float* input, float* output, size_t count) {
  for (size_t index = 0; index < count; index += 4)
    vst1q_f32(output + index, ReuseMlasExpVector(vld1q_f32(input + index)));
}
