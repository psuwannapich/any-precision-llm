// anyPrecisionKernels.cu
//
// Host launchers for the Any-Precision kernels. The device kernels themselves
// are reused verbatim from the upstream extension headers (matmul.cuh pulls in
// dequant.cuh), so there is exactly one implementation of the quantized-matmul
// math shared between the PyTorch path and this TensorRT plugin.
//
// The launch heuristics here are a faithful copy of main.cu's matmul_kbit /
// dequant_kbit host functions, minus the PyTorch tensor plumbing.

#include "anyPrecisionKernels.h"

// Upstream device kernels. CMake adds .../any_precision/modules/kernels to the
// include path so these resolve. matmul.cuh #includes dequant.cuh, which also
// provides `num_rows` (=4) and DIV_ROUND_UP.
#include "matmul.cuh"
#include <atomic>

#include <cstdio>
#include <cstdlib>

namespace anyprec
{

// ---- function-pointer dispatch tables (mirrors main.cu) ---------------------

typedef void (*matmul_func)(const __half*, const uint32_t*, const uint32_t,
                            const uint32_t, const uint32_t, const __half*,
                            __half*);
typedef void (*dequant_func)(const uint32_t*, const uint32_t, const uint32_t,
                             const __half*, __half*);

template <int s, int e>
struct get_matmul_func
{
    void operator()(matmul_func func[][9][2]) const
    {
        if constexpr (s <= e)
        {
            func[s][1][0] = matmul_kbit_32<1, s, false>;
            func[s][1][1] = matmul_kbit_32<1, s, true>;
            func[s][2][0] = matmul_kbit_32<2, s, false>;
            func[s][3][0] = matmul_kbit_32<3, s, false>;
            func[s][4][0] = matmul_kbit_32<4, s, false>;
            func[s][5][0] = matmul_kbit_32<5, s, false>;
            func[s][6][0] = matmul_kbit_32<6, s, false>;
            func[s][7][0] = matmul_kbit_32<7, s, false>;
            func[s][8][0] = matmul_kbit_32<8, s, false>;
            get_matmul_func<s + 1, e>()(func);
        }
    }
};

template <int s, int e>
struct get_dequant_func
{
    void operator()(dequant_func func[]) const
    {
        if constexpr (s <= e)
        {
            func[s] = dequant_kbit_store<s>;
            get_dequant_func<s + 1, e>()(func);
        }
    }
};

static matmul_func g_matmul[9][9][2] = {{{nullptr}}};
static dequant_func g_dequant[9] = {nullptr};
static bool g_tablesInit = false;

static void ensureTables()
{
    if (!g_tablesInit)
    {
        get_matmul_func<3, 8>()(g_matmul);
        get_dequant_func<3, 8>()(g_dequant);
        g_tablesInit = true;
    }
}

// ---- launchers --------------------------------------------------------------

void launchMatmulKbit(const __half* in, const uint32_t* qw, const __half* lut,
                      int M, int N, int K, int wBits, bool isOrin,
                      __half* out, cudaStream_t stream)
{
    ensureTables();
    const int multi_row = (M == 1 ? 1 : 4);
    // ksplit is a decode-latency optimization (M==1, K>4096, wBits>=7). It is
    // disabled by default pending validation on sm_120 (Blackwell), where it was
    // implicated in an intermittent misaligned-address fault; the non-ksplit
    // kernel is numerically identical. Set ANYPREC_KSPLIT=1 to re-enable.
    static const bool ksplitEnabled = (std::getenv("ANYPREC_KSPLIT") != nullptr);
    const int use_ksplit =
        (ksplitEnabled && !isOrin && M == 1 && K > 4096 && wBits >= 7) ? 1 : 0;
    const int num_ksplit = use_ksplit ? DIV_ROUND_UP(K, 4096) : 1;

    dim3 grid(N / (num_rows * multi_row));
    dim3 block(32, num_rows, num_ksplit);
    g_matmul[wBits][M][use_ksplit]<<<grid, block, 0, stream>>>(
        in, qw, (uint32_t) M, (uint32_t) N, (uint32_t) K, lut, out);
}

void launchDequantKbit(const uint32_t* qw, const __half* lut,
                       int N, int K, int wBits,
                       __half* wOut, cudaStream_t stream)
{
    ensureTables();
    dim3 grid(N / num_rows);
    dim3 block(32, num_rows);
    g_dequant[wBits]<<<grid, block, 0, stream>>>(
        qw, (uint32_t) N, (uint32_t) K, lut, wOut);
}

void launchDequantGemm(cublasHandle_t handle, const __half* in,
                       const uint32_t* qw, const __half* lut,
                       int M, int N, int K, int wBits,
                       __half* scratch, __half* out, cudaStream_t stream)
{
    // 1. dequantize first wBits planes -> scratch [N, K] fp16 (row-major)
    launchDequantKbit(qw, lut, N, K, wBits, scratch, stream);

    // 2. out[M,N] = in[M,K] @ scratch[N,K]^T   (matches x @ weight.T)
    //
    // cuBLAS is column-major. Treat the row-major buffers as their transposes:
    //   out^T (N x M, col-major, ld=N) = scratch^T(N x K) @ in(K x M)
    // i.e. cublas m=N, n=M, k=K, opA=T on scratch(ld=K), opB=N on in(ld=K).
    cublasSetStream(handle, stream);
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(handle,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 scratch, CUDA_R_16F, K,
                 in, CUDA_R_16F, K,
                 &beta,
                 out, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ---- bias ------------------------------------------------------------------

__global__ void addBiasKernel(__half* out, const __half* bias, int M, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;
    if (idx < total)
        out[idx] = __hadd(out[idx], bias[idx % N]);
}

void launchAddBias(__half* out, const __half* bias, int M, int N,
                   cudaStream_t stream)
{
    if (bias == nullptr)
        return;
    const int total = M * N;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    addBiasKernel<<<blocks, threads, 0, stream>>>(out, bias, M, N);
}

// ---- process-global active precision ---------------------------------------

// Process-global (NOT thread_local): TensorRT-LLM calls enqueue() on a worker
// thread, while ap_set_precision() is called from the Python/main thread. A
// thread_local here would leave the worker at its default (0) and silently run
// every request at the baked default precision.
static std::atomic<int> g_currentPrecision{0};

void setCurrentPrecision(int bits) { g_currentPrecision.store(bits, std::memory_order_release); }
int getCurrentPrecision() { return g_currentPrecision.load(std::memory_order_acquire); }

} // namespace anyprec
