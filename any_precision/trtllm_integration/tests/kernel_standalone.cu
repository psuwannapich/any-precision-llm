// kernel_standalone.cu — isolate the any-precision matmul kernel from TensorRT.
// Allocates correctly-sized, zeroed buffers for real Qwen3-4B linear shapes and
// launches matmul_kbit_32 exactly as launchMatmulKbit does, synchronizing and
// checking for CUDA errors after each launch. A fault here means the kernel
// itself is out of bounds for that (M,N,K,bits); no TRT involved.
//
// Build:
//   nvcc -O2 -std=c++17 -arch=sm_120 \
//     -I any_precision/modules/kernels \
//     any_precision/trtllm_integration/tests/kernel_standalone.cu -o /tmp/kst
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>
#include "matmul.cuh"

typedef void (*matmul_func)(const __half*, const uint32_t*, const uint32_t,
                            const uint32_t, const uint32_t, const __half*, __half*);

static matmul_func g_matmul[9][9][2] = {{{nullptr}}};
template <int s, int e> struct fill {
    void operator()() const {
        if constexpr (s <= e) {
            g_matmul[s][1][0] = matmul_kbit_32<1, s, false>;
            g_matmul[s][2][0] = matmul_kbit_32<2, s, false>;
            g_matmul[s][3][0] = matmul_kbit_32<3, s, false>;
            g_matmul[s][4][0] = matmul_kbit_32<4, s, false>;
            g_matmul[s][5][0] = matmul_kbit_32<5, s, false>;
            g_matmul[s][6][0] = matmul_kbit_32<6, s, false>;
            g_matmul[s][7][0] = matmul_kbit_32<7, s, false>;
            g_matmul[s][8][0] = matmul_kbit_32<8, s, false>;
            fill<s + 1, e>()();
        }
    }
};

static int lutTotal(int N, std::vector<int> bits) {
    int t = 0; for (int b : bits) t += N * (1 << b); return t;
}
static int lutOffset(int N, std::vector<int> bits, int p) {
    int off = 0; for (int b : bits) { if (b == p) return off; off += N * (1 << b); } return 0;
}

static bool run(int M, int N, int K, int p, std::vector<int> sup) {
    const int multi_row = (M == 1 ? 1 : 4);
    __half *in, *out, *lut; uint32_t* qw;
    int pb = 8;
    size_t qwN = (size_t)pb * N * (K / 32);
    size_t lutN = lutTotal(N, sup);
    cudaMalloc(&in, (size_t)M * K * sizeof(__half));
    cudaMalloc(&out, (size_t)M * N * sizeof(__half));
    cudaMalloc(&qw, qwN * sizeof(uint32_t));
    cudaMalloc(&lut, lutN * sizeof(__half));
    cudaMemset(in, 0, (size_t)M * K * sizeof(__half));
    cudaMemset(out, 0, (size_t)M * N * sizeof(__half));
    cudaMemset(qw, 0, qwN * sizeof(uint32_t));
    cudaMemset(lut, 0, lutN * sizeof(__half));
    cudaDeviceSynchronize();

    dim3 grid(N / (num_rows * multi_row));
    dim3 block(32, num_rows, 1);
    const __half* lp = lut + lutOffset(N, sup, p);
    g_matmul[p][M][0]<<<grid, block, 0, 0>>>(in, qw, (uint32_t)M, (uint32_t)N,
                                             (uint32_t)K, lp, out);
    cudaError_t e = cudaDeviceSynchronize();
    printf("  M=%-2d N=%-5d K=%-5d bits=%d grid=%-4d  %s\n", M, N, K, p,
           N / (num_rows * multi_row), e == cudaSuccess ? "OK" : cudaGetErrorString(e));
    cudaFree(in); cudaFree(out); cudaFree(qw); cudaFree(lut);
    return e == cudaSuccess;
}

int main() {
    fill<3, 8>()();
    std::vector<int> sup = {3, 4, 5, 6, 7, 8};
    int hidden = 2560, inter = 9728, qkv = 6144, attn = 4096;
    int Ms[] = {1, 6};
    struct { const char* name; int N, K; } layers[] = {
        {"qkv ", qkv, hidden}, {"o   ", hidden, attn},
        {"gate", inter, hidden}, {"down", hidden, inter}};
    for (auto& L : layers) {
        printf("[%s]\n", L.name);
        for (int M : Ms)
            for (int p : sup)
                run(M, L.N, L.K, p, sup);
    }
    return 0;
}
