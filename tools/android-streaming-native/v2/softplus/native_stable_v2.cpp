// NVIDIA Real-time RE-USE adaptation. NSCLv1, noncommercial research/education.
// Copyright (c) 2025 NVIDIA CORPORATION. New native adaptation (c) 2026 quangng2000.
// ORT interface headers retain Microsoft's MIT license in vendor/.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <array>
#include <cstring>
#include <mutex>
#include "../inline/mlas_exp_neon.h"
void ReuseStableSoftplus(const float*, float*, size_t);

// ORT_EXPORT is empty on Android/Linux because ORT itself uses a linker export
// map. This separately loaded library must export its own two C entry points.
#ifdef _WIN32
#define REUSE_CUSTOM_OP_EXPORT __declspec(dllexport)
#else
#define REUSE_CUSTOM_OP_EXPORT __attribute__((visibility("default")))
#endif

namespace {
constexpr int64_t kChannels = 256, kState = 16, kBins = 161;
constexpr const char* kDomain = "ai.reuse.streaming";

void Require(bool value, const char* message) {
  if (!value) throw Ort::Exception(message, ORT_INVALID_ARGUMENT);
}

struct Work {
  const float *initial, *u, *delta, *b, *c, *gate, *coefficients, *skip;
  float *final, *values;
};

// One independent channel per work item. Only the frequency dimension remains
// recurrent; every one of its original 16 FP32 state components is retained.
void Channel(void* opaque, size_t h) noexcept {
  Work& w = *static_cast<Work*>(opaque);
  alignas(16) float state[kState];
  std::memcpy(state, w.initial + h * kState, sizeof(state));
  for (int64_t t = 0; t < kBins; ++t) {
    const size_t value_index = t * kChannels + h;
    const float dt = w.delta[value_index], u = w.u[value_index];
    const float* b = w.b + t * kState;
    const float* c = w.c + t * kState;
    alignas(16) float decay[kState];
    ReuseMlasExp16(dt, w.coefficients + h * kState, decay);
    // Keep update=(delta*B)*u and state=(decay*old)+update, with FP32
    // temporaries and contraction disabled. This loop can SIMD-vectorize.
    for (int64_t j = 0; j < kState; ++j) {
      const float update_first = dt * b[j];
      const float update = update_first * u;
      const float previous = decay[j] * state[j];
      state[j] = previous + update;
    }
    float sum = 0.0f;
    for (int64_t j = 0; j < kState; ++j) {
      const float product = state[j] * c[j];
      sum = sum + product;
    }
    const float skip = w.skip[h] * u;
    const float output = sum + skip;
    w.values[value_index] = output * w.gate[value_index];
  }
  std::memcpy(w.final + h * kState, state, sizeof(state));
}

struct FrequencyKernel {
  explicit FrequencyKernel(const OrtApi&, const OrtKernelInfo* info) {
    Ort::ConstKernelInfo attributes{info};
    Require(attributes.GetAttribute<int64_t>("abi_version") == 2, "Unsupported native frequency ABI");
    Require(attributes.GetAttribute<int64_t>("channels") == kChannels &&
            attributes.GetAttribute<int64_t>("state_size") == kState &&
            attributes.GetAttribute<int64_t>("frequency_bins") == kBins,
            "Unsupported native frequency topology");
    const auto parallel = attributes.GetAttribute<int64_t>("parallel");
    Require(parallel == 0 || parallel == 1, "Unsupported native frequency parallel mode");
    parallel_ = parallel != 0;
  }

  OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext context{raw};
      Require(context.GetInputCount() == 8 && context.GetOutputCount() == 2,
              "Unexpected native frequency input/output count");
      std::array<Ort::ConstValue, 8> input = {context.GetInput(0), context.GetInput(1), context.GetInput(2),
          context.GetInput(3), context.GetInput(4), context.GetInput(5), context.GetInput(6), context.GetInput(7)};
      const std::array<std::vector<int64_t>, 8> shapes = {{
          {1, kChannels, kState}, {1, kBins, kChannels}, {1, kBins, kChannels},
          {1, kBins, kState}, {1, kBins, kState}, {1, kBins, kChannels},
          {kChannels, kState}, {kChannels}}};
      std::array<const float*, 8> pointers;
      for (size_t i = 0; i < input.size(); ++i) {
        const auto info = input[i].GetTensorTypeAndShapeInfo();
        Require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && info.GetShape() == shapes[i],
                "Native frequency requires exact bounded FP32 shapes");
        pointers[i] = input[i].GetTensorData<float>();
      }
      auto final = context.GetOutput(0, shapes[0]);
      auto values = context.GetOutput(1, shapes[1]);
      Work work{pointers[0], pointers[1], pointers[2], pointers[3], pointers[4], pointers[5], pointers[6], pointers[7],
                final.GetTensorMutableData<float>(), values.GetTensorMutableData<float>()};
      if (parallel_) context.ParallelFor(Channel, kChannels, 4, &work);
      else for (size_t channel = 0; channel < kChannels; ++channel) Channel(&work, channel);
      return nullptr;
    } catch (const Ort::Exception& error) {
      return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(), error.what());
    } catch (const std::exception& error) {
      return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
    }
  }
  bool parallel_ = true;
};


// Temporal recurrence is one update per frequency/channel, with unchanged
// FP32 Mul/Mul/Mul/Add boundaries and the exact persistent state layout.
struct TemporalWork {
  const float *old_state, *u, *delta, *b, *c, *gate, *coefficients, *skip;
  float *next_state, *values;
};

void TemporalFrequency(void* opaque, size_t frequency) noexcept {
  TemporalWork& work = *static_cast<TemporalWork*>(opaque);
  const float* b = work.b + frequency * kState;
  const float* c = work.c + frequency * kState;
  for (int64_t h = 0; h < kChannels; ++h) {
    const size_t index = frequency * kChannels + h;
    const float dt = work.delta[index], u = work.u[index];
    const float* previous = work.old_state + index * kState;
    alignas(16) float decay[kState];
    ReuseMlasExp16(dt, work.coefficients + h * kState, decay);
    float* next = work.next_state + index * kState;
    for (int64_t j = 0; j < kState; ++j) {
      const float update_first = dt * b[j];
      const float update = update_first * u;
      const float decayed = decay[j] * previous[j];
      next[j] = decayed + update;
    }
    float sum = 0.0f;
    for (int64_t j = 0; j < kState; ++j) {
      const float product = next[j] * c[j];
      sum = sum + product;
    }
    const float skip = work.skip[h] * u;
    const float output = sum + skip;
    work.values[index] = output * work.gate[index];
  }
}

struct TemporalKernel {
  explicit TemporalKernel(const OrtApi&, const OrtKernelInfo* info) {
    Ort::ConstKernelInfo attributes{info};
    Require(attributes.GetAttribute<int64_t>("abi_version") == 2, "Unsupported native temporal ABI");
    Require(attributes.GetAttribute<int64_t>("channels") == kChannels &&
            attributes.GetAttribute<int64_t>("state_size") == kState &&
            attributes.GetAttribute<int64_t>("frequency_bins") == kBins,
            "Unsupported native temporal topology");
  }
  OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext context{raw};
      Require(context.GetInputCount() == 8 && context.GetOutputCount() == 2,
              "Unexpected native temporal input/output count");
      const std::array<std::vector<int64_t>, 8> shapes = {{
          {kBins, kChannels, kState}, {kBins, kChannels}, {kBins, kChannels, 1},
          {kBins, 1, kState}, {kBins, 1, kState}, {kBins, kChannels},
          {kChannels, kState}, {kChannels}}};
      std::array<const float*, 8> pointers;
      for (size_t i = 0; i < pointers.size(); ++i) {
        auto input = context.GetInput(i);
        const auto info = input.GetTensorTypeAndShapeInfo();
        Require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && info.GetShape() == shapes[i],
                "Native temporal requires exact bounded FP32 shapes");
        pointers[i] = input.GetTensorData<float>();
      }
      auto next = context.GetOutput(0, shapes[0]);
      auto values = context.GetOutput(1, shapes[1]);
      TemporalWork work{pointers[0], pointers[1], pointers[2], pointers[3], pointers[4], pointers[5], pointers[6], pointers[7],
                        next.GetTensorMutableData<float>(), values.GetTensorMutableData<float>()};
      context.ParallelFor(TemporalFrequency, kBins, 4, &work);
      return nullptr;
    } catch (const Ort::Exception& error) {
      return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(), error.what());
    } catch (const std::exception& error) {
      return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
    }
  }
};

struct TemporalOp : Ort::CustomOpBase<TemporalOp, TemporalKernel, true> {
  const char* GetName() const { return "TemporalStepV2"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 8; }
  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try { *kernel = new TemporalKernel(api, info); return nullptr; }
    catch (const Ort::Exception& error) { return api.CreateStatus(error.GetOrtErrorCode(), error.what()); }
    catch (const std::exception& error) { return api.CreateStatus(ORT_FAIL, error.what()); }
  }
  static OrtStatusPtr InferOutputShape(Ort::ShapeInferContext& context) {
    auto status = context.SetOutputShape(0, {kBins, kChannels, kState});
    if (!status.IsOK()) return status.release();
    return context.SetOutputShape(1, {kBins, kChannels}).release();
  }
};

struct FrequencyOp : Ort::CustomOpBase<FrequencyOp, FrequencyKernel, true> {
  const char* GetName() const { return "FrequencyScanV2"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 8; }
  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try { *kernel = new FrequencyKernel(api, info); return nullptr; }
    catch (const Ort::Exception& error) { return api.CreateStatus(error.GetOrtErrorCode(), error.what()); }
    catch (const std::exception& error) { return api.CreateStatus(ORT_FAIL, error.what()); }
  }
  static OrtStatusPtr InferOutputShape(Ort::ShapeInferContext& context) {
    // Parent graph's dynamic Conv/Slice shape propagation carries symbolic
    // dimensions even though this opt-in operator is deliberately fixed.
    auto status = context.SetOutputShape(0, {1, kChannels, kState});
    if (!status.IsOK()) return status.release();
    return context.SetOutputShape(1, {1, kBins, kChannels}).release();
  }
};


struct SoftplusWork { const float* input; float* output; size_t chunk; };
void SoftplusChunk(void* opaque, size_t index) noexcept {
  auto& work = *static_cast<SoftplusWork*>(opaque);
  ReuseStableSoftplus(work.input + index * work.chunk, work.output + index * work.chunk, work.chunk);
}
struct SoftplusKernel {
  explicit SoftplusKernel(const OrtApi&, const OrtKernelInfo* info) {
    Ort::ConstKernelInfo attributes{info};
    Require(attributes.GetAttribute<int64_t>("abi_version") == 2, "Unsupported stable Softplus ABI");
    layout_ = attributes.GetAttribute<int64_t>("layout");
    Require(layout_ == 0 || layout_ == 1, "Unsupported stable Softplus layout");
  }
  OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
    try {
      Ort::KernelContext context{raw};
      Require(context.GetInputCount() == 1 && context.GetOutputCount() == 1, "Unexpected stable Softplus I/O");
      auto input = context.GetInput(0);
      const auto shape = layout_ == 0 ? std::vector<int64_t>{1,kBins,kChannels} : std::vector<int64_t>{kBins,1,kChannels};
      const auto info = input.GetTensorTypeAndShapeInfo();
      Require(info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && info.GetShape() == shape,
              "Stable Softplus requires bounded FP32 shape");
      auto output = context.GetOutput(0, shape);
      SoftplusWork work{input.GetTensorData<float>(), output.GetTensorMutableData<float>(), kBins*kChannels/4};
      context.ParallelFor(SoftplusChunk, 4, 4, &work);
      return nullptr;
    } catch (const Ort::Exception& error) { return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(), error.what()); }
    catch (const std::exception& error) { return Ort::GetApi().CreateStatus(ORT_FAIL, error.what()); }
  }
  int64_t layout_;
};
struct SoftplusOp : Ort::CustomOpBase<SoftplusOp, SoftplusKernel, true> {
  const char* GetName() const { return "StableSoftplusV2"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 1; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** kernel) const noexcept {
    try { *kernel = new SoftplusKernel(api, info); return nullptr; }
    catch (const Ort::Exception& error) { return api.CreateStatus(error.GetOrtErrorCode(), error.what()); }
    catch (const std::exception& error) { return api.CreateStatus(ORT_FAIL, error.what()); }
  }
  static OrtStatusPtr InferOutputShape(Ort::ShapeInferContext& context) {
    return context.SetOutputShape(0, context.GetInputShape(0)).release();
  }
};

FrequencyOp operation;
SoftplusOp softplus_operation;
TemporalOp temporal_operation;
std::mutex registration_mutex;
OrtCustomOpDomain* domain = nullptr;
}  // namespace

extern "C" REUSE_CUSTOM_OP_EXPORT int ORT_API_CALL ReuseStreamingKernelAbiVersion() { return 2; }

extern "C" REUSE_CUSTOM_OP_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* base) {
  const OrtApi* api = base->GetApi(ORT_API_VERSION);
  if (!api) return base->GetApi(1)->CreateStatus(ORT_FAIL, "Native scan requires ORT API23 / runtime1.23.2");
  if (std::strcmp(base->GetVersionString(), "1.23.2") != 0)
    return api->CreateStatus(ORT_FAIL, "Native scan is validated only for ONNX Runtime1.23.2");
  try {
    std::lock_guard<std::mutex> lock(registration_mutex);
    if (!domain) {
      // Published before any kernel exists. Reinitializing the global C++ API
      // pointer for a second session would race an existing ComputeV2 call.
      Ort::InitApi(api);
      Ort::ThrowOnError(api->CreateCustomOpDomain(kDomain, &domain));
      Ort::ThrowOnError(api->CustomOpDomain_Add(domain, &operation));
      Ort::ThrowOnError(api->CustomOpDomain_Add(domain, &temporal_operation));
      Ort::ThrowOnError(api->CustomOpDomain_Add(domain, &softplus_operation));
      // Deliberately process-lifetime: options and multiple sessions can retain
      // the domain. No custom API destructor can run after ORT unloads.
    } else {
      Require(&Ort::GetApi() == api, "Native scan cannot mix multiple ONNX Runtime instances");
    }
    return api->AddCustomOpDomain(options, domain);
  } catch (const Ort::Exception& error) { return api->CreateStatus(error.GetOrtErrorCode(), error.what()); }
  catch (const std::exception& error) { return api->CreateStatus(ORT_FAIL, error.what()); }
}
