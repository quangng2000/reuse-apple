cmake_minimum_required(VERSION 3.22.1)
project(reusestreaming_dsp LANGUAGES CXX)

add_library(reusestreaming_dsp SHARED reuse_audio_dsp.cpp)
target_compile_features(reusestreaming_dsp PRIVATE cxx_std_17)
target_compile_definitions(reusestreaming_dsp PRIVATE POCKETFFT_NO_MULTITHREADING=1)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # PyTorch's ARM PocketFFT baseline uses FP32 FMA within expressions. Turning
    # contraction off changes nearly-real STFT phase branches despite tiny FFT
    # error. `on` permits FMA without fast-math reassociation or reduced precision.
    target_compile_options(reusestreaming_dsp PRIVATE -O2 -fno-fast-math -ffp-contract=on)
endif()
if(NOT ANDROID)
    find_package(JNI REQUIRED)
    target_include_directories(reusestreaming_dsp PRIVATE ${JNI_INCLUDE_DIRS})
endif()
