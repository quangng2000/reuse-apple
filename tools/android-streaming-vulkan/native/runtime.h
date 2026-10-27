// SPDX-License-Identifier: MIT
// Original native runtime adapter; see LICENSE.txt in this directory.
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace reuse::vulkan {
struct Port { uint32_t offset, count; };
struct StatePair { uint32_t input, output, count; };
struct Operation { std::array<uint32_t, 32> words{}; uint32_t addressMask{}; };
struct Plan {
    uint32_t arenaFloats{};
    std::vector<Port> inputs, outputs;
    std::vector<StatePair> states;
    std::vector<Operation> operations;
    static Plan load(const std::string& path);
    uint32_t remap(uint32_t offset, unsigned bank) const;
};

// Full-network Vulkan compute. Frame output is provisional until commit();
// all recurrent tensors stay in two disjoint GPU banks across calls.
class Engine {
public:
    Engine(const std::string& planPath, const std::string& arenaPath,
           const std::string& shaderDirectory, bool allowSoftwareDevice = false,
           bool tiledMatmul = false, bool tiledConv = false, bool profileDispatches = false);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    void frame(const float* magnitude, const float* phase, float* outMagnitude,
               float* outPhase, size_t bins);
    void commit();
    void reset();
    std::vector<float> copyState(size_t index);
    std::vector<float> copyProposedState(size_t index);
    std::string infoJson() const;
    // Empty unless explicit per-dispatch profiling has produced a successful frame.
    std::string profileJson() const;
    double lastGpuMilliseconds() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace reuse::vulkan
