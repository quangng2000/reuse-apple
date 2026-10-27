#include <Eigen/Core>
#include <cstddef>
#include <cmath>
namespace {
inline float ScalarSoftplus(float x) {
  // Same selected branch as ORT, including finite/subnormal/nonfinite edges.
  return x > 0.0f ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
}
}
void ReuseStableSoftplus(const float* input, float* output, size_t count) {
  using namespace Eigen::internal;
  using Packet = Packet4f;
  size_t index = 0;
  const auto zero = pset1<Packet>(0.0f);
  for (; index + 4 <= count; index += 4) {
    const auto x = ploadu<Packet>(input + index);
    if (!(vmaxvq_f32(pabs(x)) <= 80.0f)) {
      for (size_t lane = 0; lane < 4; ++lane) output[index + lane] = ScalarSoftplus(input[index + lane]);
      continue;
    }
    const auto exp = pexp(pnegate(pabs(x)));
    // Eigen NEON has HasLog1p=0, so ordinary .log1p() silently scalarizes the
    // full expression. This pinned helper explicitly performs the corrected
    // packet log1p, including tiny-input preservation, instead of log(1+x).
    const auto result = padd(pmax(x, zero), generic_log1p<Packet>(exp));
    pstoreu<float>(output + index, result);
  }
  for (; index < count; ++index) {
    output[index] = ScalarSoftplus(input[index]);
  }
}
extern "C" __attribute__((visibility("default"))) void ReuseSoftplusArrayForValidation(const float* input, float* output, size_t count) {
  ReuseStableSoftplus(input, output, count);
}
