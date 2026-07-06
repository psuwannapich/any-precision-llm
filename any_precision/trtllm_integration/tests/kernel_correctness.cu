// kernel_correctness.cu — run matmul_kbit_32 (from matmul.cuh, plain nvcc) on the
// real dumped data and compare to the matmul_kbit reference. Tells us whether the
// kernel as-compiled-outside-torch is correct.
//   nvcc -O2 -std=c++17 -arch=sm_120 --expt-relaxed-constexpr \
//     -I any_precision/modules/kernels \
//     any_precision/trtllm_integration/tests/kernel_correctness.cu -o /tmp/kc && /tmp/kc
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

template <int M, int B>
static void launch(const __half* in, const uint32_t* qw, const __half* lut,
                   int N, int K, __half* out) {
    fill<3, 8>()();
    const int multi_row = (M == 1 ? 1 : 4);
    dim3 grid(N / (num_rows * multi_row));
    dim3 block(32, num_rows, 1);
    const char* mode = getenv("USE_FNPTR");
    if (mode) {
        g_matmul[B][M][0]<<<grid, block>>>(in, qw, (uint32_t)M, (uint32_t)N,
                                           (uint32_t)K, lut, out);
    } else {
        matmul_kbit_32<M, B, false><<<grid, block>>>(in, qw, (uint32_t)M, (uint32_t)N,
                                                     (uint32_t)K, lut, out);
    }
}

template <typename T>
static std::vector<T> load(const char* path, size_t n) {
    std::vector<T> v(n);
    FILE* f = fopen(path, "rb");
    fread(v.data(), sizeof(T), n, f);
    fclose(f);
    return v;
}

int main() {
    const int M = 4, N = 2560, K = 9728, B = 8, PB = 8;
    auto qw = load<int32_t>("/tmp/apk/qw.bin", (size_t)PB * N * (K / 32));
    auto lut = load<uint16_t>("/tmp/apk/lut.bin", (size_t)N * (1 << B));
    auto x = load<uint16_t>("/tmp/apk/x.bin", (size_t)M * K);
    auto yref = load<uint16_t>("/tmp/apk/y.bin", (size_t)M * N);

    __half *din, *dlut, *dout; uint32_t* dqw;
    cudaMalloc(&din, x.size() * 2); cudaMalloc(&dlut, lut.size() * 2);
    cudaMalloc(&dout, (size_t)M * N * 2); cudaMalloc(&dqw, qw.size() * 4);
    cudaMemcpy(din, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dlut, lut.data(), lut.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dqw, qw.data(), qw.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, (size_t)M * N * 2);

    launch<M, B>(din, (const uint32_t*)dqw, dlut, N, K, dout);
    cudaError_t e = cudaDeviceSynchronize();
    printf("launch: %s\n", cudaGetErrorString(e));

    std::vector<uint16_t> yout((size_t)M * N);
    cudaMemcpy(yout.data(), dout, yout.size() * 2, cudaMemcpyDeviceToHost);

    auto h2f = [](uint16_t h) { __half x; memcpy(&x, &h, 2); return __half2float(x); };
    float maxd = 0;
    for (size_t i = 0; i < yout.size(); i++) {
        float dd = h2f(yout[i]) - h2f(yref[i]);
        if (dd < 0) dd = -dd;
        if (dd > maxd) maxd = dd;
    }
    printf("y_ref[0,:5] = %.4f %.4f %.4f %.4f %.4f\n",
           h2f(yref[0]), h2f(yref[1]), h2f(yref[2]), h2f(yref[3]), h2f(yref[4]));
    printf("y_out[0,:5] = %.4f %.4f %.4f %.4f %.4f\n",
           h2f(yout[0]), h2f(yout[1]), h2f(yout[2]), h2f(yout[3]), h2f(yout[4]));
    printf("max abs diff = %.5f  -> %s\n", maxd,
           maxd < 0.5 ? "PASS (kernel correct)" : "FAIL (kernel wrong)");
    return 0;
}
