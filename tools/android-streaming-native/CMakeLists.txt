cmake_minimum_required(VERSION 3.22)
project(reuse_streaming_scan LANGUAGES CXX)
add_library(reuse_streaming_scan SHARED frequency_scan.cpp)
target_compile_features(reuse_streaming_scan PRIVATE cxx_std_17)
target_include_directories(reuse_streaming_scan PRIVATE vendor/onnxruntime-1.23.2)
# Preserve each FP32 multiplication/addition boundary; no fast-math or FMA fusion.
target_compile_options(reuse_streaming_scan PRIVATE -O3 -fno-fast-math -ffp-contract=off -fvisibility=hidden)
