// SPDX-License-Identifier: MIT
// Original native runtime adapter; see LICENSE.txt in this directory.
#include "runtime.h"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace reuse::vulkan {
namespace {
constexpr uint32_t kBins = 321;
constexpr uint64_t kMaximumArenaBytes = 1024ull * 1024 * 1024;
constexpr uint32_t kMaximumOperations = 20000;
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
void vkCheck(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(operation) + " failed with Vulkan status " + std::to_string(result));
}
std::vector<uint8_t> readFile(const std::string& path, uint64_t maximum) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    require(bool(stream), "Cannot open " + path);
    const auto end = stream.tellg();
    require(end >= 0 && uint64_t(end) <= maximum, "File size is invalid: " + path);
    std::vector<uint8_t> bytes(static_cast<size_t>(end));
    stream.seekg(0);
    if (!bytes.empty()) stream.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
    require(bool(stream), "Truncated file: " + path);
    return bytes;
}
uint32_t get32(const std::vector<uint8_t>& bytes, size_t& cursor) {
    require(cursor <= bytes.size() && bytes.size() - cursor >= 4, "Truncated Vulkan plan");
    uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= uint32_t(bytes[cursor++]) << (8 * i);
    return value;
}
bool overlaps(uint32_t a, uint32_t an, uint32_t b, uint32_t bn) {
    return uint64_t(a) < uint64_t(b) + bn && uint64_t(b) < uint64_t(a) + an;
}
std::string escape(const std::string& text) {
    std::ostringstream stream;
    for (unsigned char c : text) {
        if (c == '\\' || c == '"') stream << '\\' << c;
        else if (c < 32) stream << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else stream << c;
    }
    return stream.str();
}
bool dense3x3Shape(const std::array<uint32_t, 32>& p) {
    return p[22] == 1 && p[24] == 0 && p[25] == 0 && p[26] == 0 && p[7] >= 16 &&
           p[8] == 1 && p[14] == 3 && p[15] == 3 && p[16] == 1 && p[17] == 1 &&
           p[18] == 0 && p[19] == 1 && p[21] == 1 &&
           uint64_t(p[12]) == 1ull + 2ull * p[20] && p[9] == p[13];
}
const char* shaderName(uint32_t opcode) {
    switch (opcode) {
        case 1: return "unary";
        case 2: return "binary";
        case 3: return "where";
        case 4: return "gather";
        case 5: return "pad";
        case 6: return "scatter";
        case 7: return "matmul";
        case 8: return "conv";
        case 9: return "layernorm";
        case 10: return "frequency";
        case 11: return "temporal";
        case 13: return "concat_cache";
        case 107: return "matmul_tiled";
        case 108: return "conv_tiled";
        case 208: return "conv_3x3_tiled";
        default: throw std::runtime_error("Unsupported Vulkan opcode " + std::to_string(opcode));
    }
}
} // namespace

Plan Plan::load(const std::string& path) {
    auto bytes = readFile(path, 16 * 1024 * 1024);
    require(bytes.size() >= 32 && std::memcmp(bytes.data(), "RVKPLAN1", 8) == 0, "Invalid Vulkan plan magic");
    size_t cursor = 8;
    require(get32(bytes, cursor) == 1, "Unsupported Vulkan plan version");
    Plan plan;
    plan.arenaFloats = get32(bytes, cursor);
    const uint32_t operationCount = get32(bytes, cursor);
    const uint32_t inputCount = get32(bytes, cursor);
    const uint32_t outputCount = get32(bytes, cursor);
    const uint32_t stateCount = get32(bytes, cursor);
    require(plan.arenaFloats > 0 && uint64_t(plan.arenaFloats) * 4 <= kMaximumArenaBytes, "Invalid arena size");
    require(operationCount > 0 && operationCount <= kMaximumOperations, "Invalid operation count");
    require(inputCount == 33 && outputCount == 33 && stateCount == 31, "Expected eight-layer input/output/state inventory");
    const uint64_t expected = 32ull + (uint64_t(inputCount) + outputCount) * 8 + uint64_t(stateCount) * 12 + uint64_t(operationCount) * 132;
    require(expected == bytes.size(), "Vulkan plan size does not match its record counts");
    const auto validSpan = [&](uint32_t offset, uint32_t count) {
        require(count > 0 && uint64_t(offset) + count <= plan.arenaFloats, "Tensor span exceeds Vulkan arena");
    };
    const auto checkedSpan = [&](uint32_t offset, uint64_t count) {
        require(count > 0 && count <= plan.arenaFloats && uint64_t(offset) + count <= plan.arenaFloats,
                "Operation tensor extent exceeds Vulkan arena");
    };
    const auto product = [&](std::initializer_list<uint32_t> dimensions) {
        uint64_t value = 1;
        for (auto d : dimensions) {
            require(d > 0 && value <= plan.arenaFloats / d, "Invalid/overflowing operation shape");
            value *= d;
        }
        return value;
    };
    for (unsigned which = 0; which < 2; ++which) {
        auto& ports = which == 0 ? plan.inputs : plan.outputs;
        const uint32_t count = which == 0 ? inputCount : outputCount;
        for (uint32_t i = 0; i < count; ++i) {
            Port port{get32(bytes, cursor), get32(bytes, cursor)};
            validSpan(port.offset, port.count);
            for (const auto previous : ports) require(!overlaps(port.offset, port.count, previous.offset, previous.count), "Graph ports overlap");
            ports.push_back(port);
        }
        require(ports[0].count == kBins && ports[1].count == kBins, "Expected 321-bin magnitude and phase");
    }
    for (const auto input : plan.inputs) for (const auto output : plan.outputs) require(!overlaps(input.offset, input.count, output.offset, output.count), "Input and output banks overlap");
    std::vector<Port> stateSpans;
    for (uint32_t i = 0; i < stateCount; ++i) {
        StatePair state{get32(bytes, cursor), get32(bytes, cursor), get32(bytes, cursor)};
        validSpan(state.input, state.count);
        validSpan(state.output, state.count);
        require(state.input == plan.inputs[i + 2].offset && state.output == plan.outputs[i + 2].offset &&
                state.count == plan.inputs[i + 2].count && state.count == plan.outputs[i + 2].count,
                "State mappings do not match graph ports");
        for (const auto span : {Port{state.input, state.count}, Port{state.output, state.count}}) {
            for (const auto prior : stateSpans) require(!overlaps(span.offset, span.count, prior.offset, prior.count), "State banks overlap");
            for (unsigned j = 0; j < 2; ++j) {
                require(!overlaps(span.offset, span.count, plan.inputs[j].offset, plan.inputs[j].count), "State overlaps input spectrum");
                require(!overlaps(span.offset, span.count, plan.outputs[j].offset, plan.outputs[j].count), "State overlaps output spectrum");
            }
            stateSpans.push_back(span);
        }
        plan.states.push_back(state);
    }
    for (uint32_t i = 0; i < operationCount; ++i) {
        Operation op;
        for (auto& word : op.words) word = get32(bytes, cursor);
        op.addressMask = get32(bytes, cursor);
        require((op.words[0] >= 1 && op.words[0] <= 11) || op.words[0] == 13, "Unsupported plan opcode");
        shaderName(op.words[0]); // Reject unsupported code before any GPU allocation.
        require((op.addressMask & 1u) == 0 && (op.addressMask & 4u) == 0, "Invalid operation address mask");
        require(op.addressMask & 2u, "Operation output address is not marked");
        validSpan(op.words[1], op.words[2]);
        for (unsigned j = 0; j < 32; ++j) {
            if (op.addressMask & (1u << j)) require(op.words[j] < plan.arenaFloats, "Operation input address exceeds arena");
        }
        const auto& p = op.words;
        const auto strided = [&](uint32_t base, unsigned strideStart) {
            int64_t minimum = base, maximum = base;
            uint64_t count = 1;
            for (unsigned axis = 0; axis < 6; ++axis) {
                require(p[8 + axis] > 0 && count <= plan.arenaFloats / p[8 + axis], "Invalid strided operation shape");
                count *= p[8 + axis];
                const int64_t extent = int64_t(int32_t(p[strideStart + axis])) * (p[8 + axis] - 1);
                minimum += std::min<int64_t>(0, extent); maximum += std::max<int64_t>(0, extent);
            }
            require(count == p[2] && minimum >= 0 && maximum < plan.arenaFloats, "Strided operation exceeds arena or output shape");
            return Port{uint32_t(minimum), uint32_t(maximum - minimum + 1)};
        };
        uint32_t requiredMask = 0xau, allowedMask = 0xau;
        Port writeSpan{p[1], p[2]};
        switch (p[0]) {
            case 1:
                require(p[6] <= 14, "Unsupported unary operation");
                checkedSpan(p[3], p[2]); break;
            case 2:
                require(p[6] >= 1 && p[6] <= 11, "Unsupported binary operation");
                requiredMask = allowedMask = 0x1au;
                strided(p[3], 14); strided(p[4], 20); break;
            case 3:
                requiredMask = allowedMask = 0x3au;
                strided(p[3], 14); strided(p[4], 20); strided(p[5], 26); break;
            case 4: strided(p[3], 14); break;
            case 5: {
                uint64_t inputElements = 1, outputElements = 1;
                for (unsigned axis = 0; axis < 6; ++axis) {
                    require(p[8 + axis] > 0 && p[14 + axis] > 0 && inputElements <= plan.arenaFloats / p[14 + axis] && outputElements <= plan.arenaFloats / p[8 + axis], "Invalid pad shape");
                    inputElements *= p[14 + axis]; outputElements *= p[8 + axis];
                }
                require(outputElements == p[2], "Pad output count mismatch");
                checkedSpan(p[3], inputElements); break;
            }
            case 6:
                checkedSpan(p[3], p[2]); writeSpan = strided(p[1], 14); break;
            case 7:
                requiredMask = 0x1au; allowedMask = 0x3au;
                require(p[12] <= 1 && p[13] <= 1, "Invalid matrix transpose flag");
                require(product({p[6], p[7], p[9]}) == p[2], "Matrix output shape mismatch");
                checkedSpan(p[3], product({p[6], p[8]}) + uint64_t(p[9] - 1) * p[10]);
                checkedSpan(p[4], product({p[8], p[7]}) + uint64_t(p[9] - 1) * p[11]);
                require(p[14] <= 2 && p[15] <= 1, "Invalid matrix fused addition/activation mode");
                if (p[14]) { requiredMask |= 0x20u; checkedSpan(p[5], p[14] == 1 ? p[7] : p[2]); }
                break;
            case 8: {
                requiredMask = 0x1au; allowedMask = 0x3au;
                require(p[22] > 0 && p[7] % p[22] == 0 && p[11] % p[22] == 0, "Invalid convolution groups");
                require(p[16] > 0 && p[17] > 0 && p[20] > 0 && p[21] > 0 && p[23] <= 1 && p[24] <= 1 && p[25] <= 1 && p[26] <= 1, "Invalid convolution attributes");
                require(p[27] <= 1 && (!p[27] || dense3x3Shape(p)), "Invalid dense 3x3 specialization hint");
                require(product({p[6], p[7], p[8], p[9]}) == p[2], "Convolution output shape mismatch");
                checkedSpan(p[3], product({p[6], p[11], p[12], p[13]}));
                checkedSpan(p[4], product({p[7], p[11] / p[22], p[14], p[15]}));
                if (p[23]) { requiredMask |= 0x20u; checkedSpan(p[5], p[7]); }
                break;
            }
            case 9:
                allowedMask = 0x403au;
                require(p[9] <= 1 && p[10] <= 1 && product({p[6], p[7]}) == p[2], "Invalid layernorm shape/flags");
                checkedSpan(p[3], p[2]);
                if (p[9]) { requiredMask |= 0x10u; checkedSpan(p[4], p[7]); }
                if (p[10]) { requiredMask |= 0x20u; checkedSpan(p[5], p[7]); }
                { float epsilon; std::memcpy(&epsilon, &p[8], 4); require(std::isfinite(epsilon) && epsilon > 0, "Invalid layernorm epsilon"); }
                require(p[11] <= 1 && p[13] <= 1, "Invalid layernorm fused layout/activation mode");
                if (p[11]) require(p[12] > 0 && p[6] % p[12] == 0, "Invalid channel-first layernorm layout");
                if (p[13]) {
                    require(p[15] == 1 || p[15] == p[7], "Invalid fused PReLU slope count");
                    requiredMask |= 0x4000u; checkedSpan(p[14], p[15]);
                }
                break;
            case 10:
            case 11: {
                requiredMask = allowedMask = 0x7fau;
                require(p[14] == 16 && product({p[12], p[13]}) == p[2], "Invalid recurrent operator shape");
                const uint64_t states = p[0] == 10 ? product({p[13], 16}) : product({p[12], p[13], 16});
                checkedSpan(p[3], states);
                for (unsigned j : {4u, 5u, 8u}) checkedSpan(p[j], p[2]);
                checkedSpan(p[6], product({p[12], 16})); checkedSpan(p[7], product({p[12], 16}));
                checkedSpan(p[9], product({p[13], 16})); checkedSpan(p[10], p[13]);
                if (p[11] != UINT32_MAX) {
                    requiredMask |= 0x800u; allowedMask |= 0x800u; checkedSpan(p[11], states);
                    for (const auto port : plan.inputs) require(!overlaps(p[11], uint32_t(states), port.offset, port.count), "Recurrent output overwrites committed state");
                } else require(p[0] == 10, "Temporal state output is mandatory");
                break;
            }
            case 13: {
                requiredMask = allowedMask = 0x1au;
                require(p[6] > 0 && p[7] > 0 && p[8] > 0 && p[10] <= 1,
                        "Invalid fused concat shape/cache flag");
                const uint64_t inner = uint64_t(p[6]) + p[7];
                require(inner <= plan.arenaFloats && product({uint32_t(inner), p[8]}) == p[2],
                        "Fused concat output shape mismatch");
                const auto aCount = uint32_t(product({p[6], p[8]}));
                const auto bCount = uint32_t(product({p[7], p[8]}));
                checkedSpan(p[3], aCount); checkedSpan(p[4], bCount);
                for (const auto source : {Port{p[3], aCount}, Port{p[4], bCount}})
                    require(!overlaps(p[1], p[2], source.offset, source.count),
                            "Fused concat output aliases source");
                for (unsigned j = 11; j < 32; ++j)
                    require(p[j] == 0, "Nonzero reserved fused concat field");
                if (p[10]) {
                    requiredMask = allowedMask = 0x3au;
                    require(p[9] > 0 && p[9] < inner && p[9] == p[7], "Invalid fused cache crop");
                    const auto cacheCount = uint32_t(product({uint32_t(inner) - p[9], p[8]}));
                    checkedSpan(p[5], cacheCount);
                    require(!overlaps(p[5], cacheCount, p[1], p[2]), "Fused cache aliases concat output");
                    for (const auto source : {Port{p[3], aCount}, Port{p[4], bCount}})
                        require(!overlaps(p[5], cacheCount, source.offset, source.count),
                                "Fused cache output aliases source");
                    bool exactState = false;
                    for (const auto state : plan.states)
                        if (state.input == p[3] && state.output == p[5] &&
                            state.count == cacheCount && state.count == aCount) exactState = true;
                    require(exactState, "Fused cache does not match a complete next-state port");
                    for (const auto port : plan.inputs)
                        require(!overlaps(p[5], cacheCount, port.offset, port.count),
                                "Fused cache overwrites committed state");
                } else {
                    require(p[5] == 0 && p[9] == 0, "Cache fields set without fused cache output");
                }
                break;
            }
        }
        require((op.addressMask & requiredMask) == requiredMask && (op.addressMask & ~allowedMask) == 0,
                "Operation address mask does not match its opcode");
        for (const auto port : plan.inputs) require(!overlaps(writeSpan.offset, writeSpan.count, port.offset, port.count), "Operation overwrites graph input or committed state");
        plan.operations.push_back(op);
    }
    require(cursor == bytes.size(), "Trailing Vulkan plan bytes");
    return plan;
}

uint32_t Plan::remap(uint32_t offset, unsigned bank) const {
    if (bank == 0) return offset;
    for (const auto& state : states) {
        if (offset >= state.input && uint64_t(offset) < uint64_t(state.input) + state.count) return state.output + (offset - state.input);
        if (offset >= state.output && uint64_t(offset) < uint64_t(state.output) + state.count) return state.input + (offset - state.output);
    }
    return offset;
}

struct Engine::Impl {
    struct Buffer {
        VkBuffer handle = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        VkDeviceSize allocationSize = 0;
        VkMemoryPropertyFlags flags = 0;
        void* mapped = nullptr;
    };
    Plan plan;
    mutable std::mutex mutex;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceFloatControlsProperties floats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    uint32_t timestampBits = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    std::map<uint32_t, VkPipeline> pipelines;
    VkPipeline finitePipeline = VK_NULL_HANDLE;
    VkCommandBuffer commands[2]{VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkQueryPool queryPool = VK_NULL_HANDLE;
    Buffer arena, input, output, status;
    unsigned currentBank = 0;
    bool pending = false;
    bool poisoned = false;
    bool allowSoftware = false;
    bool tiledMatmul = false, tiledConv = false;
    bool profileDispatches = false;
    uint32_t queryCount = 2;
    uint64_t profiledFrameOrdinal = 0;
    std::vector<uint64_t> lastProfileTicks;
    double gpuMilliseconds = -1;
    std::string shaderDirectory;

    Impl(const std::string& planPath, const std::string& arenaPath, const std::string& shaderPath, bool software, bool tileMatmul, bool tileConv, bool profile)
        : plan(Plan::load(planPath)), allowSoftware(software), tiledMatmul(tileMatmul), tiledConv(tileConv), profileDispatches(profile), shaderDirectory(shaderPath) {
        try {
            const auto initial = readFile(arenaPath, kMaximumArenaBytes);
            require(initial.size() == uint64_t(plan.arenaFloats) * 4, "Arena file size does not match plan");
            for (size_t i = 0; i < initial.size(); i += 4) {
                float value;
                std::memcpy(&value, initial.data() + i, 4);
                require(std::isfinite(value), "Initial Vulkan arena contains nonfinite values");
            }
            initializeDevice();
            require(initial.size() <= properties.limits.maxStorageBufferRange, "Model arena exceeds GPU maxStorageBufferRange");
            require(properties.limits.maxPushConstantsSize >= 128, "GPU lacks 128-byte push constants");
            require(properties.limits.maxComputeWorkGroupInvocations >= 256 && properties.limits.maxComputeWorkGroupSize[0] >= 256 && properties.limits.maxComputeWorkGroupSize[1] >= 8, "GPU compute workgroup limits are insufficient");
            arena = makeBuffer(initial.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            input = makeBuffer(kBins * 2 * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            output = makeBuffer(kBins * 2 * 4 + 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            status = makeBuffer(4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            uploadArena(initial);
            initializePipelines();
            recordCommands(0);
            recordCommands(1);
        } catch (...) { destroy(); throw; }
    }
    ~Impl() { destroy(); }
    void destroyBuffer(Buffer& buffer) {
        if (!device) return;
        if (buffer.mapped) vkUnmapMemory(device, buffer.memory);
        if (buffer.handle) vkDestroyBuffer(device, buffer.handle, nullptr);
        if (buffer.memory) vkFreeMemory(device, buffer.memory, nullptr);
        buffer = {};
    }
    void destroy() noexcept {
        if (device) {
            vkDeviceWaitIdle(device);
            for (auto entry : pipelines) vkDestroyPipeline(device, entry.second, nullptr);
            pipelines.clear();
            if (finitePipeline) vkDestroyPipeline(device, finitePipeline, nullptr);
            if (queryPool) vkDestroyQueryPool(device, queryPool, nullptr);
            if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            if (descriptorLayout) vkDestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
            destroyBuffer(arena); destroyBuffer(input); destroyBuffer(output); destroyBuffer(status);
            vkDestroyDevice(device, nullptr);
            device = VK_NULL_HANDLE;
        }
        if (instance) vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
    void initializeDevice() {
        uint32_t extensionCount = 0;
        vkCheck(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr), "enumerate instance extensions");
        std::vector<VkExtensionProperties> available(extensionCount);
        vkCheck(vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, available.data()), "read instance extensions");
        std::vector<const char*> extensions;
        bool portability = false;
        for (const auto& entry : available) if (std::strcmp(entry.extensionName, "VK_KHR_portability_enumeration") == 0) portability = true;
        if (portability) extensions.push_back("VK_KHR_portability_enumeration");
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "ReuseStreaming FP32 Vulkan";
        app.applicationVersion = 1;
        app.pEngineName = "Reuse native compute";
        app.engineVersion = 1;
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        create.pApplicationInfo = &app;
        create.enabledExtensionCount = uint32_t(extensions.size());
        create.ppEnabledExtensionNames = extensions.data();
        if (portability) create.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        vkCheck(vkCreateInstance(&create, nullptr, &instance), "create Vulkan instance");
        uint32_t count = 0;
        vkCheck(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate GPUs");
        require(count > 0, "No Vulkan device found");
        std::vector<VkPhysicalDevice> candidates(count);
        vkCheck(vkEnumeratePhysicalDevices(instance, &count, candidates.data()), "read GPUs");
        for (auto candidate : candidates) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(candidate, &props);
            const bool gpu = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            if (!gpu && !allowSoftware) continue;
            if (props.apiVersion < VK_API_VERSION_1_1) continue;
            uint32_t families = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, nullptr);
            std::vector<VkQueueFamilyProperties> familyProperties(families);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &families, familyProperties.data());
            for (uint32_t i = 0; i < families; ++i) {
                if (!(familyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) || !familyProperties[i].queueCount) continue;
                physical = candidate; properties = props; queueFamily = i; timestampBits = familyProperties[i].timestampValidBits;
                break;
            }
            if (physical) break;
        }
        require(physical != VK_NULL_HANDLE, "No supported hardware Vulkan GPU/compute queue; CPU fallback is disabled");
        VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties2.pNext = &subgroup;
        subgroup.pNext = &floats;
        const auto getProperties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
        require(getProperties2 != nullptr, "Vulkan 1.1 properties entry point is unavailable");
        getProperties2(physical, &properties2);
        require(subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT, "GPU lacks compute subgroups");
        require(subgroup.subgroupSize >= 16 && subgroup.subgroupSize % 16 == 0, "GPU subgroup size must be a multiple of sixteen");
        const VkSubgroupFeatureFlags neededSubgroups = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
        require((subgroup.supportedOperations & neededSubgroups) == neededSubgroups, "GPU lacks required subgroup shuffle operations");
        vkGetPhysicalDeviceMemoryProperties(physical, &memoryProperties);
        uint32_t deviceExtensionCount = 0;
        vkCheck(vkEnumerateDeviceExtensionProperties(physical, nullptr, &deviceExtensionCount, nullptr), "enumerate device extensions");
        std::vector<VkExtensionProperties> deviceExtensions(deviceExtensionCount);
        vkCheck(vkEnumerateDeviceExtensionProperties(physical, nullptr, &deviceExtensionCount, deviceExtensions.data()), "read device extensions");
        std::vector<const char*> enabled;
        for (const auto& entry : deviceExtensions) if (std::strcmp(entry.extensionName, "VK_KHR_portability_subset") == 0) enabled.push_back("VK_KHR_portability_subset");
        const float priority = 1.f;
        VkDeviceQueueCreateInfo queueCreate{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueCreate.queueFamilyIndex = queueFamily; queueCreate.queueCount = 1; queueCreate.pQueuePriorities = &priority;
        VkPhysicalDeviceFeatures availableFeatures{};
        vkGetPhysicalDeviceFeatures(physical, &availableFeatures);
        VkPhysicalDeviceFeatures features{};
        features.robustBufferAccess = availableFeatures.robustBufferAccess;
        VkDeviceCreateInfo deviceCreate{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceCreate.queueCreateInfoCount = 1; deviceCreate.pQueueCreateInfos = &queueCreate;
        deviceCreate.enabledExtensionCount = uint32_t(enabled.size()); deviceCreate.ppEnabledExtensionNames = enabled.data();
        deviceCreate.pEnabledFeatures = &features;
        vkCheck(vkCreateDevice(physical, &deviceCreate, nullptr, &device), "create Vulkan device");
        vkGetDeviceQueue(device, queueFamily, 0, &queue);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = queueFamily; pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCheck(vkCreateCommandPool(device, &pool, nullptr, &commandPool), "create command pool");
        VkFenceCreateInfo fenceCreate{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCheck(vkCreateFence(device, &fenceCreate, nullptr, &fence), "create fence");
        require(!profileDispatches || timestampBits > 0, "Selected GPU queue does not support timestamp profiling");
        queryCount = profileDispatches ? uint32_t(plan.operations.size()) * 2 + 7 : 2;
        if (timestampBits > 0) {
            VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            query.queryType = VK_QUERY_TYPE_TIMESTAMP; query.queryCount = queryCount;
            vkCheck(vkCreateQueryPool(device, &query, nullptr, &queryPool), "create timestamp pool");
        }
    }
    Buffer makeBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags required) {
        Buffer result;
        try {
            result.size = size;
            VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            create.size = size; create.usage = usage; create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            vkCheck(vkCreateBuffer(device, &create, nullptr, &result.handle), "create buffer");
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device, result.handle, &requirements);
            uint32_t selected = UINT32_MAX;
            unsigned bestScore = 0;
            for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
                const auto flags = memoryProperties.memoryTypes[i].propertyFlags;
                if (!(requirements.memoryTypeBits & (1u << i)) || (flags & required) != required) continue;
                const unsigned score = 1 + ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? 2 : 0) + ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? 1 : 0);
                if (selected == UINT32_MAX || score > bestScore) { selected = i; bestScore = score; }
            }
            require(selected != UINT32_MAX, "GPU has no compatible memory type");
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size; allocation.memoryTypeIndex = selected;
            vkCheck(vkAllocateMemory(device, &allocation, nullptr, &result.memory), "allocate GPU memory");
            result.allocationSize = requirements.size;
            result.flags = memoryProperties.memoryTypes[selected].propertyFlags;
            vkCheck(vkBindBufferMemory(device, result.handle, result.memory, 0), "bind GPU memory");
            if (required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) vkCheck(vkMapMemory(device, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped), "map staging memory");
            return result;
        } catch (...) { destroyBuffer(result); throw; }
    }
    void flush(const Buffer& buffer) {
        if (buffer.flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) return;
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = buffer.memory; range.size = VK_WHOLE_SIZE;
        vkCheck(vkFlushMappedMemoryRanges(device, 1, &range), "flush staging memory");
    }
    void invalidate(const Buffer& buffer) {
        if (buffer.flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) return;
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        range.memory = buffer.memory; range.size = VK_WHOLE_SIZE;
        vkCheck(vkInvalidateMappedMemoryRanges(device, 1, &range), "invalidate result memory");
    }
    VkCommandBuffer allocateCommand() {
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = commandPool; allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocation.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        vkCheck(vkAllocateCommandBuffers(device, &allocation, &command), "allocate command buffer");
        return command;
    }
    void submit(VkCommandBuffer command) {
        require(!poisoned, "Vulkan engine is unusable after a device/submission failure");
        try {
            vkCheck(vkResetFences(device, 1, &fence), "reset fence");
            VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submitInfo.commandBufferCount = 1; submitInfo.pCommandBuffers = &command;
            vkCheck(vkQueueSubmit(queue, 1, &submitInfo, fence), "submit inference");
            vkCheck(vkWaitForFences(device, 1, &fence, VK_TRUE, 30ull * 1000 * 1000 * 1000), "wait for GPU inference");
        } catch (...) { poisoned = true; pending = false; throw; }
    }
    static void barrier(VkCommandBuffer command, VkPipelineStageFlags source, VkPipelineStageFlags destination, VkAccessFlags reads, VkAccessFlags writes) {
        VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        memory.srcAccessMask = reads; memory.dstAccessMask = writes;
        vkCmdPipelineBarrier(command, source, destination, 0, 1, &memory, 0, nullptr, 0, nullptr);
    }
    void uploadArena(const std::vector<uint8_t>& bytes) {
        Buffer staging = makeBuffer(bytes.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        VkCommandBuffer command = VK_NULL_HANDLE;
        try {
            std::memcpy(staging.mapped, bytes.data(), bytes.size()); flush(staging);
            command = allocateCommand();
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkCheck(vkBeginCommandBuffer(command, &begin), "begin upload");
            VkBufferCopy copy{0, 0, bytes.size()}; vkCmdCopyBuffer(command, staging.handle, arena.handle, 1, &copy);
            barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            vkCheck(vkEndCommandBuffer(command), "end upload"); submit(command);
            vkFreeCommandBuffers(device, commandPool, 1, &command); destroyBuffer(staging);
        } catch (...) {
            if (command) vkFreeCommandBuffers(device, commandPool, 1, &command);
            destroyBuffer(staging); throw;
        }
    }
    VkPipeline loadPipeline(const std::string& name) {
        const auto bytes = readFile(shaderDirectory + "/" + name + ".spv", 8 * 1024 * 1024);
        require(bytes.size() >= 20 && bytes.size() % 4 == 0, "Invalid SPIR-V size: " + name);
        std::vector<uint32_t> code(bytes.size() / 4);
        std::memcpy(code.data(), bytes.data(), bytes.size());
        require(code[0] == 0x07230203u, "Invalid SPIR-V magic: " + name);
        VkShaderModuleCreateInfo moduleCreate{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleCreate.codeSize = bytes.size(); moduleCreate.pCode = code.data();
        VkShaderModule module = VK_NULL_HANDLE;
        vkCheck(vkCreateShaderModule(device, &moduleCreate, nullptr, &module), "create shader module");
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; stage.module = module; stage.pName = "main";
        // Candidate kernels can statically choose a subgroup-width path without
        // relaxed-precision builtins. Shaders without SpecId 0 ignore this map.
        const VkSpecializationMapEntry subgroupEntry{0, 0, sizeof(uint32_t)};
        const VkSpecializationInfo subgroupSpecialization{1, &subgroupEntry, sizeof(uint32_t), &subgroup.subgroupSize};
        stage.pSpecializationInfo = &subgroupSpecialization;
        VkComputePipelineCreateInfo create{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        create.stage = stage; create.layout = pipelineLayout;
        const auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &create, nullptr, &pipeline);
        vkDestroyShaderModule(device, module, nullptr);
        vkCheck(result, ("create compute pipeline " + name).c_str());
        return pipeline;
    }
    uint32_t pipelineKey(const std::array<uint32_t, 32>& words) const {
        if (words[0] == 7 && tiledMatmul) return 107;
        if (words[0] == 8 && tiledConv && words[27] == 1) return 208;
        if (words[0] == 8 && tiledConv && words[22] == 1 && words[24] == 0 && words[25] == 0 && words[26] == 0 && words[7] >= 16) return 108;
        return words[0];
    }
    void initializePipelines() {
        VkDescriptorSetLayoutBinding bindings[2]{};
        for (unsigned i = 0; i < 2; ++i) { bindings[i].binding = i; bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; bindings[i].descriptorCount = 1; bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.bindingCount = 2; layout.pBindings = bindings;
        vkCheck(vkCreateDescriptorSetLayout(device, &layout, nullptr, &descriptorLayout), "create descriptor layout");
        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1; pool.poolSizeCount = 1; pool.pPoolSizes = &poolSize;
        vkCheck(vkCreateDescriptorPool(device, &pool, nullptr, &descriptorPool), "create descriptor pool");
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = descriptorPool; allocate.descriptorSetCount = 1; allocate.pSetLayouts = &descriptorLayout;
        vkCheck(vkAllocateDescriptorSets(device, &allocate, &descriptorSet), "allocate descriptor set");
        VkDescriptorBufferInfo info[2]{{arena.handle, 0, arena.size}, {status.handle, 0, status.size}};
        VkWriteDescriptorSet writes[2]{};
        for (unsigned i = 0; i < 2; ++i) { writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = descriptorSet; writes[i].dstBinding = i; writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &info[i]; }
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
        VkPipelineLayoutCreateInfo pipelineCreate{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineCreate.setLayoutCount = 1; pipelineCreate.pSetLayouts = &descriptorLayout; pipelineCreate.pushConstantRangeCount = 1; pipelineCreate.pPushConstantRanges = &range;
        vkCheck(vkCreatePipelineLayout(device, &pipelineCreate, nullptr, &pipelineLayout), "create pipeline layout");
        for (const auto& op : plan.operations) { const auto key = pipelineKey(op.words); if (pipelines.find(key) == pipelines.end()) pipelines[key] = loadPipeline(shaderName(key)); }
        finitePipeline = loadPipeline("finite_check");
    }
    std::array<uint32_t, 3> dispatch(const std::array<uint32_t, 32>& words) const;
    void recordCommands(unsigned bank) {
        commands[bank] = allocateCommand();
        auto command = commands[bank];
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkCheck(vkBeginCommandBuffer(command, &begin), "begin inference commands");
        if (queryPool) { vkCmdResetQueryPool(command, queryPool, 0, queryCount); vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0); }
        vkCmdFillBuffer(command, status.handle, 0, 4, 0);
        VkBufferCopy inputCopies[2]{{0, uint64_t(plan.inputs[0].offset) * 4, kBins * 4}, {kBins * 4, uint64_t(plan.inputs[1].offset) * 4, kBins * 4}};
        vkCmdCopyBuffer(command, input.handle, arena.handle, 2, inputCopies);
        if (profileDispatches) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TRANSFER_BIT, queryPool, 1);
        barrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        VkPipeline previous = VK_NULL_HANDLE;
        uint32_t operationIndex = 0;
        if (profileDispatches) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 2);
        for (const auto& op : plan.operations) {
            auto words = op.words;
            for (unsigned i = 0; i < 32; ++i) if (op.addressMask & (1u << i)) words[i] = plan.remap(words[i], bank);
            const auto groups = dispatch(words);
            for (unsigned i = 0; i < 3; ++i) require(groups[i] > 0 && groups[i] <= properties.limits.maxComputeWorkGroupCount[i], "Dispatch exceeds GPU group limits");
            const auto pipeline = pipelines.at(pipelineKey(words));
            if (pipeline != previous) { vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline); previous = pipeline; }
            vkCmdPushConstants(command, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 128, words.data());
            if (profileDispatches) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 3 + operationIndex * 2);
            vkCmdDispatch(command, groups[0], groups[1], groups[2]);
            if (profileDispatches) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 4 + operationIndex * 2);
            ++operationIndex;
            barrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
        const uint32_t graphEndQuery = uint32_t(plan.operations.size()) * 2 + 3;
        if (profileDispatches) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, graphEndQuery);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, finitePipeline);
        for (const auto port : plan.outputs) {
            std::array<uint32_t, 32> words{}; words[0] = plan.remap(port.offset, bank); words[1] = port.count;
            vkCmdPushConstants(command, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 128, words.data());
            vkCmdDispatch(command, (port.count + 127) / 128, 1, 1);
        }
        if (profileDispatches) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, graphEndQuery + 1);
        barrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferCopy outputCopies[2]{{uint64_t(plan.outputs[0].offset) * 4, 0, kBins * 4}, {uint64_t(plan.outputs[1].offset) * 4, kBins * 4, kBins * 4}};
        vkCmdCopyBuffer(command, arena.handle, output.handle, 2, outputCopies);
        VkBufferCopy flagCopy{0, kBins * 2 * 4, 4}; vkCmdCopyBuffer(command, status.handle, output.handle, 1, &flagCopy);
        if (profileDispatches) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TRANSFER_BIT, queryPool, graphEndQuery + 2);
        barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        if (queryPool) vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, queryCount - 1);
        vkCheck(vkEndCommandBuffer(command), "end inference commands");
    }
    void frame(const float* magnitude, const float* phase, float* outMagnitude, float* outPhase, size_t bins) {
        std::lock_guard<std::mutex> guard(mutex);
        pending = false;
        require(bins == kBins, "Vulkan network requires 321 bins");
        require(magnitude && phase && outMagnitude && outPhase, "Null frame array");
        for (size_t i = 0; i < bins; ++i) require(std::isfinite(magnitude[i]) && std::isfinite(phase[i]), "Nonfinite input spectrum");
        pending = false;
        auto* staging = static_cast<float*>(input.mapped);
        std::memcpy(staging, magnitude, bins * 4); std::memcpy(staging + bins, phase, bins * 4); flush(input);
        submit(commands[currentBank]);
        invalidate(output);
        const auto* result = static_cast<const float*>(output.mapped);
        uint32_t nonfinite = 0; std::memcpy(&nonfinite, result + 2 * bins, 4);
        require(nonfinite == 0, "Vulkan frame contains nonfinite output or recurrent state; state was not committed");
        std::memcpy(outMagnitude, result, bins * 4); std::memcpy(outPhase, result + bins, bins * 4);
        pending = true;
        if (queryPool && !profileDispatches) {
            // Keep the normal timing path at its original two stack-allocated queries.
            uint64_t ticks[2]{};
            const auto result = vkGetQueryPoolResults(device, queryPool, 0, 2, sizeof(ticks), ticks,
                sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (result == VK_SUCCESS) gpuMilliseconds = milliseconds(ticks[0], ticks[1]);
        } else if (queryPool) {
            std::vector<uint64_t> ticks(queryCount);
            const auto queryResult = vkGetQueryPoolResults(device, queryPool, 0, queryCount,
                ticks.size() * sizeof(uint64_t), ticks.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (queryResult == VK_SUCCESS) {
                gpuMilliseconds = milliseconds(ticks.front(), ticks.back());
                lastProfileTicks = std::move(ticks); ++profiledFrameOrdinal;
            } else {
                pending = false;
                vkCheck(queryResult, "read per-dispatch GPU timestamps");
            }
        }
    }
    void commit() {
        std::lock_guard<std::mutex> guard(mutex);
        require(!poisoned && pending, "No successful provisional Vulkan frame to commit");
        currentBank = 1 - currentBank;
        pending = false;
    }
    void reset() {
        std::lock_guard<std::mutex> guard(mutex);
        auto command = allocateCommand();
        try {
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkCheck(vkBeginCommandBuffer(command, &begin), "begin reset");
            for (const auto state : plan.states) {
                vkCmdFillBuffer(command, arena.handle, uint64_t(state.input) * 4, uint64_t(state.count) * 4, 0);
                vkCmdFillBuffer(command, arena.handle, uint64_t(state.output) * 4, uint64_t(state.count) * 4, 0);
            }
            barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            vkCheck(vkEndCommandBuffer(command), "end reset"); submit(command);
            vkFreeCommandBuffers(device, commandPool, 1, &command);
            currentBank = 0; pending = false; gpuMilliseconds = -1;
            lastProfileTicks.clear(); profiledFrameOrdinal = 0;
        } catch (...) { vkFreeCommandBuffers(device, commandPool, 1, &command); throw; }
    }
    std::vector<float> copyState(size_t index, bool proposed) {
        std::lock_guard<std::mutex> guard(mutex);
        require(index < plan.states.size(), "State index out of range");
        require(!proposed || pending, "No proposed state to read");
        const auto& state = plan.states[index];
        const auto offset = plan.remap(proposed ? state.output : state.input, currentBank);
        Buffer staging = makeBuffer(uint64_t(state.count) * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        auto command = allocateCommand();
        try {
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkCheck(vkBeginCommandBuffer(command, &begin), "begin state readback");
            barrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferCopy copy{uint64_t(offset) * 4, 0, uint64_t(state.count) * 4}; vkCmdCopyBuffer(command, arena.handle, staging.handle, 1, &copy);
            barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            vkCheck(vkEndCommandBuffer(command), "end state readback"); submit(command); invalidate(staging);
            std::vector<float> result(state.count); std::memcpy(result.data(), staging.mapped, result.size() * 4);
            vkFreeCommandBuffers(device, commandPool, 1, &command); destroyBuffer(staging); return result;
        } catch (...) { vkFreeCommandBuffers(device, commandPool, 1, &command); destroyBuffer(staging); throw; }
    }
    double milliseconds(uint64_t from, uint64_t to) const {
        const uint64_t mask = timestampBits == 64 ? UINT64_MAX : ((uint64_t(1) << timestampBits) - 1);
        return double((to - from) & mask) * properties.limits.timestampPeriod / 1e6;
    }
    std::string profileJson() const {
        std::lock_guard<std::mutex> guard(mutex);
        if (!profileDispatches || lastProfileTicks.size() != queryCount) return {};
        const auto interval = [&](size_t from, size_t to) { return milliseconds(lastProfileTicks.at(from), lastProfileTicks.at(to)); };
        const size_t graphEnd = plan.operations.size() * 2 + 3;
        double dispatchTotal = 0;
        std::map<std::string, std::pair<uint32_t, double>> categories;
        for (size_t i = 0; i < plan.operations.size(); ++i) {
            const auto name = std::string(shaderName(pipelineKey(plan.operations[i].words)));
            const double duration = interval(3 + i * 2, 4 + i * 2);
            dispatchTotal += duration;
            auto& category = categories[name]; ++category.first; category.second += duration;
        }
        std::ostringstream out; out << std::setprecision(12);
        out << "{\"schemaVersion\":1,\"profiled\":true,\"frameOrdinal\":" << profiledFrameOrdinal
            << ",\"queryCount\":" << queryCount << ",\"timestampValidBits\":" << timestampBits
            << ",\"timestampPeriodNs\":" << properties.limits.timestampPeriod
            << ",\"operationCount\":" << plan.operations.size()
            << ",\"wholeGpuMs\":" << interval(0, queryCount - 1)
            << ",\"inputAndStatusInitMs\":" << interval(0, 1)
            << ",\"inputDependencyMs\":" << interval(1, 2)
            << ",\"computeGraphMs\":" << interval(2, graphEnd)
            << ",\"summedDispatchMs\":" << dispatchTotal
            << ",\"unattributedGraphMs\":" << interval(2, graphEnd) - dispatchTotal
            << ",\"finiteValidationMs\":" << interval(graphEnd, graphEnd + 1)
            << ",\"outputReadbackMs\":" << interval(graphEnd + 1, graphEnd + 2)
            << ",\"completionMs\":" << interval(graphEnd + 2, graphEnd + 3)
            << ",\"timingNote\":\"Per-dispatch timestamps perturb scheduling. Graph residual includes dependency waits and timestamp overhead; it is not isolated barrier cost. Compare separate unprofiled public-push timings.\",\"categories\":[";
        bool first = true;
        for (const auto& category : categories) {
            out << (first ? "" : ",") << "{\"kernel\":\"" << category.first << "\",\"dispatches\":" << category.second.first << ",\"gpuMs\":" << category.second.second << "}";
            first = false;
        }
        out << "],\"operations\":[";
        for (size_t i = 0; i < plan.operations.size(); ++i) {
            const auto& words = plan.operations[i].words; const auto groups = dispatch(words);
            out << (i ? "," : "") << "{\"index\":" << i << ",\"opcode\":" << words[0]
                << ",\"kernel\":\"" << shaderName(pipelineKey(words)) << "\",\"outputElements\":" << words[2]
                << ",\"workgroups\":[" << groups[0] << "," << groups[1] << "," << groups[2] << "]"
                << ",\"gpuMs\":" << interval(3 + i * 2, 4 + i * 2) << "}";
        }
        out << "]}";
        return out.str();
    }
    std::string infoJson() const {
        std::ostringstream out;
        out << "{\"backend\":\"native-cpp-vulkan-fp32\",\"deviceName\":\"" << escape(properties.deviceName)
            << "\",\"vendorId\":" << properties.vendorID << ",\"deviceId\":" << properties.deviceID
            << ",\"deviceType\":" << properties.deviceType << ",\"apiVersion\":" << properties.apiVersion
            << ",\"driverVersion\":" << properties.driverVersion << ",\"subgroupSize\":" << subgroup.subgroupSize
            << ",\"subgroupOperations\":" << subgroup.supportedOperations
            << ",\"hardwareGpu\":" << ((properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? "true" : "false")
            << ",\"softwareDeviceAllowed\":" << (allowSoftware ? "true" : "false")
            << ",\"perDispatchProfiling\":" << (profileDispatches ? "true" : "false")
            << ",\"timestampValidBits\":" << timestampBits
            << ",\"arenaBytes\":" << arena.size << ",\"operations\":" << plan.operations.size()
            << ",\"stateCount\":" << plan.states.size() << ",\"precision\":\"float32\",\"statePingPong\":true,\"gpuFiniteChecks\":true"
            << ",\"tiledMatmulPipeline\":" << (pipelines.count(107) ? "true" : "false")
            << ",\"tiledConvolutionPipeline\":" << (pipelines.count(108) ? "true" : "false")
            << ",\"float32DenormPreserveSupported\":" << (floats.shaderDenormPreserveFloat32 ? "true" : "false")
            << ",\"maxStorageBufferRange\":" << properties.limits.maxStorageBufferRange << "}";
        return out.str();
    }
};

// Shader ABI dispatch mapping is finalized alongside the exporter and shaders.
std::array<uint32_t, 3> Engine::Impl::dispatch(const std::array<uint32_t, 32>& words) const {
    switch (words[0]) {
        case 7: if (tiledMatmul) return {(words[7] + 31) / 32, (words[6] + 15) / 16, words[9]}; else return {(words[7] + 15) / 16, (words[6] + 7) / 8, words[9]};
        case 8: if (pipelineKey(words) == 108 || pipelineKey(words) == 208) return {(words[8] * words[9] + 31) / 32, (words[7] + 15) / 16, words[6]}; else return {(words[2] + 127) / 128, 1, 1};
        case 9: return {words[6], 1, 1};
        case 10: return {(words[13] + 3) / 4, 1, 1};
        case 11: return {(words[2] + 3) / 4, 1, 1};
        default: return {(words[2] + 127) / 128, 1, 1};
    }
}

Engine::Engine(const std::string& plan, const std::string& arena, const std::string& shaders, bool allowSoftware, bool tiledMatmul, bool tiledConv, bool profileDispatches)
    : impl_(std::make_unique<Impl>(plan, arena, shaders, allowSoftware, tiledMatmul, tiledConv, profileDispatches)) {}
Engine::~Engine() = default;
void Engine::frame(const float* magnitude, const float* phase, float* outMagnitude, float* outPhase, size_t bins) { impl_->frame(magnitude, phase, outMagnitude, outPhase, bins); }
void Engine::commit() { impl_->commit(); }
void Engine::reset() { impl_->reset(); }
std::vector<float> Engine::copyState(size_t index) { return impl_->copyState(index, false); }
std::vector<float> Engine::copyProposedState(size_t index) { return impl_->copyState(index, true); }
std::string Engine::infoJson() const { return impl_->infoJson(); }
std::string Engine::profileJson() const { return impl_->profileJson(); }
double Engine::lastGpuMilliseconds() const { return impl_->gpuMilliseconds; }
} // namespace reuse::vulkan
