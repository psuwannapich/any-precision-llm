// anyPrecisionKernels.h
//
// Host-side launchers for the Any-Precision LUT kernels, exposed so the
// TensorRT plugin (anyPrecisionPlugin.cpp) can call them without pulling in
// the PyTorch extension (main.cu).
//
// The actual device kernels live in the upstream headers
//   any_precision/modules/kernels/matmul.cuh   (matmul_kbit_32)
//   any_precision/modules/kernels/dequant.cuh  (dequant_kbit_store)
// which this translation unit #includes, so there is a single source of
// truth for the quantized-matmul math.
//
// Weight layout (identical to AnyPrecisionLinear):
//   qweight : int32   [parent_bits, N, K/32]  bit-plane packed; precision p
//                                              uses planes [0:p].
//   lut_b   : float16 [N, 2^b]                 per-output-row centroids for
//                                              each supported bit-width b.
//   bias    : float16 [N]                      optional.
//
// Two execution paths, mirroring AnyPrecisionLinear.forward:
//   M (= rows of activations) <= 8 : fused matmul_kbit  (reads packed bits)
//   M > 8                          : dequant_kbit -> cuBLAS HGEMM
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace anyprec
{

// Fused quantized GEMV/GEMM path (M <= 8).
//   in   : [M, K] fp16
//   qw   : [parent_bits, N, K/32] int32  (only first w_bits planes are read)
//   lut  : [N, 2^w_bits] fp16
//   out  : [M, N] fp16
// `isOrin` disables the K-split heuristic on Jetson Orin (matches main.cu).
void launchMatmulKbit(const __half* in, const uint32_t* qw, const __half* lut,
                      int M, int N, int K, int wBits, bool isOrin,
                      __half* out, cudaStream_t stream);

// Dequantize the first `wBits` planes of qw into a dense [N, K] fp16 tensor.
//   wOut : [N, K] fp16  (caller-provided scratch)
void launchDequantKbit(const uint32_t* qw, const __half* lut,
                       int N, int K, int wBits,
                       __half* wOut, cudaStream_t stream);

// Dense path (M > 8): dequant into `scratch` ([N,K] fp16) then
// out[M,N] = in[M,K] * dequant(qw)^T  via cuBLAS HGEMM.
void launchDequantGemm(cublasHandle_t handle, const __half* in,
                       const uint32_t* qw, const __half* lut,
                       int M, int N, int K, int wBits,
                       __half* scratch, __half* out, cudaStream_t stream);

// Add bias[N] to out[M,N] in-place (no-op if bias == nullptr).
void launchAddBias(__half* out, const __half* bias, int M, int N,
                   cudaStream_t stream);

// Bytes of scratch the dense path needs for a given problem size.
inline size_t dequantScratchBytes(int N, int K)
{
    return static_cast<size_t>(N) * static_cast<size_t>(K) * sizeof(__half);
}

// Process-global active precision, mirroring the vLLM precision_manager design.
// The runtime sets this before each decode/prefill step; every AnyPrecision
// plugin instance reads it inside enqueue(). 0 means "unset" -> the plugin
// falls back to its baked-in default precision. Thread-local so concurrent
// engines on different host threads do not clobber each other.
void setCurrentPrecision(int bits);
int getCurrentPrecision();

} // namespace anyprec
