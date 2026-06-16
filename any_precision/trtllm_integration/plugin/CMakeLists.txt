# CMakeLists.txt — builds libanyprec_trt_plugin.so
#
# The plugin reuses the upstream device kernels in
#   ../../modules/kernels/{matmul.cuh,dequant.cuh}
# so there is a single source of truth for the quantized-matmul math.
#
# Requirements (see ../README.md):
#   * NVIDIA GPU with compute capability >= 8.0 (Ampere/Ada/Hopper).
#     The any-precision kernels target sm_70+, but TensorRT-LLM itself requires
#     sm_80+, so the engine that loads this plugin must run on sm_80+.
#   * CUDA Toolkit (cublas), TensorRT >= 9 (10 recommended).
#
# Configure example:
#   cmake -B build \
#     -DTRT_ROOT=/path/to/TensorRT \
#     -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
#   cmake --build build -j

cmake_minimum_required(VERSION 3.18)
project(anyprec_trt_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

find_package(CUDAToolkit REQUIRED)

# --- locate TensorRT -------------------------------------------------------
# Provide -DTRT_ROOT=/path/to/TensorRT, or rely on system paths.
set(TRT_ROOT "" CACHE PATH "TensorRT installation root")
find_path(TRT_INCLUDE_DIR NvInfer.h
  HINTS ${TRT_ROOT}/include /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include)
find_library(TRT_NVINFER nvinfer
  HINTS ${TRT_ROOT}/lib ${TRT_ROOT}/lib64 /usr/lib/x86_64-linux-gnu)
find_library(TRT_NVINFER_PLUGIN nvinfer_plugin
  HINTS ${TRT_ROOT}/lib ${TRT_ROOT}/lib64 /usr/lib/x86_64-linux-gnu)

if(NOT TRT_INCLUDE_DIR OR NOT TRT_NVINFER)
  message(FATAL_ERROR "TensorRT not found. Pass -DTRT_ROOT=/path/to/TensorRT")
endif()

# Upstream any-precision CUDA kernels (matmul.cuh / dequant.cuh).
set(AP_KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../modules/kernels)

add_library(anyprec_trt_plugin SHARED
  anyPrecisionPlugin.cpp
  anyPrecisionKernels.cu)

target_include_directories(anyprec_trt_plugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${AP_KERNELS_DIR}
  ${TRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS})

target_link_libraries(anyprec_trt_plugin PRIVATE
  ${TRT_NVINFER}
  ${TRT_NVINFER_PLUGIN}
  CUDA::cudart
  CUDA::cublas)

# The dequant/matmul kernels use __half intrinsics and warp shuffles.
target_compile_options(anyprec_trt_plugin PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

set_target_properties(anyprec_trt_plugin PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)
