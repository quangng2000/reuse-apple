cmake_minimum_required(VERSION 3.22)
project(reuse_streaming_stable_v2 LANGUAGES CXX)
add_library(reuse_streaming_stable_v2 SHARED native_stable_v2.cpp stable_softplus.cpp)
target_compile_features(reuse_streaming_stable_v2 PRIVATE cxx_std_17)
target_include_directories(reuse_streaming_stable_v2 PRIVATE ../../vendor/onnxruntime-1.23.2)
target_compile_options(reuse_streaming_stable_v2 PRIVATE -O3 -fno-fast-math -ffp-contract=off -fvisibility=hidden)


target_include_directories(reuse_streaming_stable_v2 PRIVATE vendor/eigen)
