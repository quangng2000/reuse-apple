// SPDX-License-Identifier: MIT
// Original native runtime adapter; see LICENSE.txt in this directory.
#include "runtime.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
std::vector<float> readFloats(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot read input " + path);
    auto length = stream.tellg();
    if (length <= 0 || uint64_t(length) % (642 * 4) != 0 || uint64_t(length) > 1024ull * 1024 * 1024) throw std::runtime_error("Input must contain 642 FP32 values per frame");
    std::vector<float> result(size_t(length) / 4);
    stream.seekg(0); stream.read(reinterpret_cast<char*>(result.data()), length);
    if (!stream) throw std::runtime_error("Input read failed");
    return result;
}
void writeFloats(const std::string& path, const std::vector<float>& values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(values.data()), std::streamsize(values.size() * 4));
    if (!stream) throw std::runtime_error("Output write failed " + path);
}
void requireIdentical(const std::vector<float>& a, const std::vector<float>& b, const char* label) {
    if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size() * 4) != 0) throw std::runtime_error(label);
}
void testTransactions(reuse::vulkan::Engine& engine, const std::vector<float>& input) {
    engine.reset();
    std::vector<float> first(642), repeated(642), second(642), afterFailure(642);
    engine.frame(input.data(), input.data() + 321, first.data(), first.data() + 321, 321);
    for (size_t i = 0; i < 31; ++i) for (float value : engine.copyState(i)) {
        if (value != 0.f) throw std::runtime_error("Provisional frame modified committed zero state");
    }
    engine.frame(input.data(), input.data() + 321, repeated.data(), repeated.data() + 321, 321);
    requireIdentical(first, repeated, "Discarded frame changed the next inference result");
    engine.commit();
    std::vector<std::vector<float>> committed;
    for (size_t i = 0; i < 31; ++i) committed.push_back(engine.copyState(i));
    const size_t next = input.size() >= 1284 ? 642 : 0;
    engine.frame(input.data() + next, input.data() + next + 321, second.data(), second.data() + 321, 321);
    auto badInput = std::vector<float>(input.begin() + next, input.begin() + next + 642);
    badInput[0] = std::numeric_limits<float>::quiet_NaN();
    bool failed = false;
    try { engine.frame(badInput.data(), badInput.data() + 321, afterFailure.data(), afterFailure.data() + 321, 321); }
    catch (const std::exception&) { failed = true; }
    if (!failed) throw std::runtime_error("Nonfinite input was accepted");
    failed = false;
    try { engine.commit(); } catch (const std::exception&) { failed = true; }
    if (!failed) throw std::runtime_error("Commit succeeded after a failed frame");
    for (size_t i = 0; i < 31; ++i) requireIdentical(committed[i], engine.copyState(i), "Failed/discarded frame changed committed state");
    engine.frame(input.data() + next, input.data() + next + 321, afterFailure.data(), afterFailure.data() + 321, 321);
    requireIdentical(second, afterFailure, "Failed frame affected the next provisional output");
    engine.commit();
    engine.reset();
    engine.frame(input.data(), input.data() + 321, repeated.data(), repeated.data() + 321, 321);
    requireIdentical(first, repeated, "Reset after transaction test did not reproduce first frame");
    engine.commit();
    engine.frame(input.data() + next, input.data() + next + 321, repeated.data(), repeated.data() + 321, 321);
    requireIdentical(second, repeated, "Transaction committed recurrent state more than once");
    engine.commit();
}
}
int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "validate-plan") {
            const auto plan = reuse::vulkan::Plan::load(argv[2]);
            std::cout << "{\"valid\":true,\"arenaBytes\":" << uint64_t(plan.arenaFloats) * 4 << ",\"operations\":" << plan.operations.size() << "}\n";
            return 0;
        }
        if (argc < 6) {
            std::cerr << "Usage: reuse-vulkan-cli PLAN WEIGHTS SHADERS INPUT.f32 OUTPUT.f32 [--states DIR] [--allow-software] [--reset-replay]\n";
            return 2;
        }
        bool allowSoftware = false, resetReplay = false, tiledMatmul = false, tiledConv = false, transactionTest = false, profile = false;
        std::string states;
        for (int i = 6; i < argc; ++i) {
            if (std::strcmp(argv[i], "--allow-software") == 0) allowSoftware = true;
            else if (std::strcmp(argv[i], "--reset-replay") == 0) resetReplay = true;
            else if (std::strcmp(argv[i], "--tiled-matmul") == 0) tiledMatmul = true;
            else if (std::strcmp(argv[i], "--tiled-conv") == 0) tiledConv = true;
            else if (std::strcmp(argv[i], "--transaction-test") == 0) transactionTest = true;
            else if (std::strcmp(argv[i], "--profile") == 0) profile = true;
            else if (std::strcmp(argv[i], "--states") == 0 && i + 1 < argc) states = argv[++i];
            else throw std::runtime_error("Unknown argument " + std::string(argv[i]));
        }
        const auto input = readFloats(argv[4]);
        const auto setupStart = std::chrono::steady_clock::now();
        reuse::vulkan::Engine engine(argv[1], argv[2], argv[3], allowSoftware, tiledMatmul, tiledConv, profile);
        const double setupMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - setupStart).count();
        std::vector<float> output(input.size());
        std::vector<double> elapsed, gpu;
        for (size_t offset = 0; offset < input.size(); offset += 642) {
            const auto start = std::chrono::steady_clock::now();
            engine.frame(input.data() + offset, input.data() + offset + 321, output.data() + offset, output.data() + offset + 321, 321);
            engine.commit();
            elapsed.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            gpu.push_back(engine.lastGpuMilliseconds());
        }
        writeFloats(argv[5], output);
        if (!states.empty()) {
            std::filesystem::create_directories(states);
            for (size_t i = 0; i < 31; ++i) writeFloats(states + "/state-" + std::to_string(i) + ".f32", engine.copyState(i));
        }
        if (resetReplay) {
            engine.reset();
            std::vector<float> replay(input.size());
            for (size_t offset = 0; offset < input.size(); offset += 642) {
                engine.frame(input.data() + offset, input.data() + offset + 321, replay.data() + offset, replay.data() + offset + 321, 321); engine.commit();
            }
            if (std::memcmp(output.data(), replay.data(), output.size() * 4) != 0) throw std::runtime_error("Reset replay is not bitwise identical");
        }
        if (transactionTest) testTransactions(engine, input);
        std::cout << "{\"device\":" << engine.infoJson() << ",\"setupMs\":" << setupMs << ",\"frames\":" << elapsed.size()
                  << ",\"resetReplayBitwise\":" << (resetReplay ? "true" : "null")
                  << ",\"transactionTestPassed\":" << (transactionTest ? "true" : "null") << ",\"frameMs\":[";
        for (size_t i = 0; i < elapsed.size(); ++i) std::cout << (i ? "," : "") << elapsed[i];
        std::cout << "],\"gpuMs\":[";
        for (size_t i = 0; i < gpu.size(); ++i) std::cout << (i ? "," : "") << gpu[i];
        std::cout << "]";
        const auto gpuProfile = engine.profileJson();
        std::cout << ",\"profile\":" << (gpuProfile.empty() ? "null" : gpuProfile) << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Vulkan error: " << error.what() << '\n';
        return 1;
    }
}
