// anyPrecisionPlugin.cpp — implementation of the AnyPrecisionLinear TRT plugin.

#include "anyPrecisionPlugin.h"

#include <cassert>
#include <cstring>
#include <cstdio>

using namespace nvinfer1;

namespace
{
// ---- POD (de)serialization helpers ----
template <typename T>
void writePod(char*& p, const T& v)
{
    std::memcpy(p, &v, sizeof(T));
    p += sizeof(T);
}
template <typename T>
T readPod(const char*& p)
{
    T v;
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
}
template <typename T>
void writeArr(char*& p, const T* src, size_t n)
{
    std::memcpy(p, src, n * sizeof(T));
    p += n * sizeof(T);
}
template <typename T>
void readArr(const char*& p, T* dst, size_t n)
{
    std::memcpy(dst, p, n * sizeof(T));
    p += n * sizeof(T);
}

#define AP_CUDA_CHECK(call)                                                     \
    do {                                                                       \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "[AnyPrecisionPlugin] CUDA error %s at %s:%d\n", \
                         cudaGetErrorString(_e), __FILE__, __LINE__);         \
            return -1;                                                        \
        }                                                                    \
    } while (0)
} // namespace

namespace anyprec
{

// ---- helpers ----------------------------------------------------------------

size_t AnyPrecisionPlugin::lutTotalElems() const
{
    size_t total = 0;
    for (int b : mSupportedBits)
        total += static_cast<size_t>(mN) * (1ULL << b);
    return total;
}

int AnyPrecisionPlugin::lutOffsetElems(int bits) const
{
    int off = 0;
    for (int b : mSupportedBits)
    {
        if (b == bits)
            return off;
        off += mN * (1 << b);
    }
    return 0; // not found -> first LUT (defensive)
}

// ---- constructors -----------------------------------------------------------

AnyPrecisionPlugin::AnyPrecisionPlugin(int N, int K, int seedBits, int parentBits,
    const std::vector<int>& supportedBits, int defaultPrecision,
    const int32_t* qweight, const __half* luts, const __half* bias)
    : mN(N), mK(K), mSeedBits(seedBits), mParentBits(parentBits),
      mDefaultPrecision(defaultPrecision), mHasBias(bias != nullptr),
      mSupportedBits(supportedBits)
{
    const size_t qwCount = static_cast<size_t>(parentBits) * N * (K / 32);
    mQweightHost.resize(qwCount);
    std::memcpy(mQweightHost.data(), qweight, qwCount * sizeof(int32_t));

    const size_t lutCount = lutTotalElems();
    mLutHost.resize(lutCount);
    std::memcpy(mLutHost.data(), luts, lutCount * sizeof(__half));

    if (mHasBias)
    {
        mBiasHost.resize(N);
        std::memcpy(mBiasHost.data(), bias, N * sizeof(__half));
    }
}

AnyPrecisionPlugin::AnyPrecisionPlugin(const void* data, size_t length)
{
    const char* p = static_cast<const char*>(data);
    mN = readPod<int>(p);
    mK = readPod<int>(p);
    mSeedBits = readPod<int>(p);
    mParentBits = readPod<int>(p);
    mDefaultPrecision = readPod<int>(p);
    mHasBias = readPod<int>(p) != 0;

    const int nSupported = readPod<int>(p);
    mSupportedBits.resize(nSupported);
    readArr(p, mSupportedBits.data(), nSupported);

    const size_t qwCount = readPod<size_t>(p);
    mQweightHost.resize(qwCount);
    readArr(p, mQweightHost.data(), qwCount);

    const size_t lutCount = readPod<size_t>(p);
    mLutHost.resize(lutCount);
    readArr(p, mLutHost.data(), lutCount);

    const size_t biasCount = readPod<size_t>(p);
    if (biasCount)
    {
        mBiasHost.resize(biasCount);
        readArr(p, mBiasHost.data(), biasCount);
    }
    assert(p == static_cast<const char*>(data) + length);
    (void) length;
}

// ---- IPluginV2DynamicExt ----------------------------------------------------

IPluginV2DynamicExt* AnyPrecisionPlugin::clone() const noexcept
{
    auto* plugin = new AnyPrecisionPlugin(mN, mK, mSeedBits, mParentBits,
        mSupportedBits, mDefaultPrecision, mQweightHost.data(), mLutHost.data(),
        mHasBias ? mBiasHost.data() : nullptr);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

DimsExprs AnyPrecisionPlugin::getOutputDimensions(int outputIndex,
    const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    // out = in with the last (hidden) dim replaced by N.
    DimsExprs out = inputs[0];
    out.d[out.nbDims - 1] = exprBuilder.constant(mN);
    return out;
}

bool AnyPrecisionPlugin::supportsFormatCombination(int pos,
    const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // single fp16/linear input, single fp16/linear output
    const auto& desc = inOut[pos];
    return desc.type == DataType::kHALF && desc.format == TensorFormat::kLINEAR;
}

void AnyPrecisionPlugin::configurePlugin(const DynamicPluginTensorDesc* in,
    int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t AnyPrecisionPlugin::getWorkspaceSize(const PluginTensorDesc* inputs,
    int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    // dense path needs a [N, K] fp16 dequant scratch buffer
    return dequantScratchBytes(mN, mK);
}

int AnyPrecisionPlugin::enqueue(const PluginTensorDesc* inputDesc,
    const PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    // M = product of all dims except the last (the hidden dim K).
    const Dims& inDims = inputDesc[0].dims;
    int M = 1;
    for (int i = 0; i < inDims.nbDims - 1; ++i)
        M *= inDims.d[i];
    const int K = inDims.d[inDims.nbDims - 1];
    const int N = mN;

    // Pick precision: process-global override, else baked default.
    int precision = getCurrentPrecision();
    if (precision == 0)
        precision = mDefaultPrecision;
    if (precision < mSeedBits || precision > mParentBits)
        precision = mDefaultPrecision;

    const auto* in = static_cast<const __half*>(inputs[0]);
    auto* out = static_cast<__half*>(outputs[0]);
    const __half* lut = mLutDev + lutOffsetElems(precision);

    if (M >= 1 && M <= 8)
    {
        launchMatmulKbit(in, reinterpret_cast<const uint32_t*>(mQweightDev), lut,
                         M, N, K, precision, mIsOrin, out, stream);
    }
    else
    {
        auto* scratch = static_cast<__half*>(workspace);
        launchDequantGemm(mCublas, in,
                          reinterpret_cast<const uint32_t*>(mQweightDev), lut,
                          M, N, K, precision, scratch, out, stream);
    }
    launchAddBias(out, mHasBias ? mBiasDev : nullptr, M, N, stream);
    return 0;
}

// ---- IPluginV2Ext -----------------------------------------------------------

DataType AnyPrecisionPlugin::getOutputDataType(int index,
    const DataType* inputTypes, int nbInputs) const noexcept
{
    return DataType::kHALF;
}

// ---- IPluginV2 --------------------------------------------------------------

int AnyPrecisionPlugin::initialize() noexcept
{
    const size_t qwBytes = mQweightHost.size() * sizeof(int32_t);
    AP_CUDA_CHECK(cudaMalloc(&mQweightDev, qwBytes));
    AP_CUDA_CHECK(cudaMemcpy(mQweightDev, mQweightHost.data(), qwBytes,
                             cudaMemcpyHostToDevice));

    const size_t lutBytes = mLutHost.size() * sizeof(__half);
    AP_CUDA_CHECK(cudaMalloc(&mLutDev, lutBytes));
    AP_CUDA_CHECK(cudaMemcpy(mLutDev, mLutHost.data(), lutBytes,
                             cudaMemcpyHostToDevice));

    if (mHasBias)
    {
        const size_t biasBytes = mBiasHost.size() * sizeof(__half);
        AP_CUDA_CHECK(cudaMalloc(&mBiasDev, biasBytes));
        AP_CUDA_CHECK(cudaMemcpy(mBiasDev, mBiasHost.data(), biasBytes,
                                 cudaMemcpyHostToDevice));
    }

    if (cublasCreate(&mCublas) != CUBLAS_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "[AnyPrecisionPlugin] cublasCreate failed\n");
        return -1;
    }

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        mIsOrin = std::strcmp(prop.name, "Orin") == 0;

    return 0;
}

void AnyPrecisionPlugin::terminate() noexcept
{
    if (mQweightDev) { cudaFree(mQweightDev); mQweightDev = nullptr; }
    if (mLutDev) { cudaFree(mLutDev); mLutDev = nullptr; }
    if (mBiasDev) { cudaFree(mBiasDev); mBiasDev = nullptr; }
    if (mCublas) { cublasDestroy(mCublas); mCublas = nullptr; }
}

size_t AnyPrecisionPlugin::getSerializationSize() const noexcept
{
    size_t sz = 0;
    sz += sizeof(int) * 6;                                   // N,K,seed,parent,default,hasBias
    sz += sizeof(int);                                       // nSupported
    sz += sizeof(int) * mSupportedBits.size();
    sz += sizeof(size_t) + mQweightHost.size() * sizeof(int32_t);
    sz += sizeof(size_t) + mLutHost.size() * sizeof(__half);
    sz += sizeof(size_t) + mBiasHost.size() * sizeof(__half);
    return sz;
}

void AnyPrecisionPlugin::serialize(void* buffer) const noexcept
{
    char* p = static_cast<char*>(buffer);
    writePod(p, mN);
    writePod(p, mK);
    writePod(p, mSeedBits);
    writePod(p, mParentBits);
    writePod(p, mDefaultPrecision);
    writePod(p, static_cast<int>(mHasBias ? 1 : 0));

    writePod(p, static_cast<int>(mSupportedBits.size()));
    writeArr(p, mSupportedBits.data(), mSupportedBits.size());

    writePod(p, mQweightHost.size());
    writeArr(p, mQweightHost.data(), mQweightHost.size());

    writePod(p, mLutHost.size());
    writeArr(p, mLutHost.data(), mLutHost.size());

    writePod(p, mBiasHost.size());
    writeArr(p, mBiasHost.data(), mBiasHost.size());
}

void AnyPrecisionPlugin::destroy() noexcept
{
    delete this;
}

// ---- Creator ----------------------------------------------------------------

PluginFieldCollection AnyPrecisionPluginCreator::mFC{};
std::vector<PluginField> AnyPrecisionPluginCreator::mFields{};

AnyPrecisionPluginCreator::AnyPrecisionPluginCreator()
{
    mFields.clear();
    mFields.emplace_back("N", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("K", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("seed_bits", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("parent_bits", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("default_precision", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("supported_bits", nullptr, PluginFieldType::kINT32, 0);
    mFields.emplace_back("qweight", nullptr, PluginFieldType::kINT32, 0);
    mFields.emplace_back("luts", nullptr, PluginFieldType::kFLOAT16, 0);
    mFields.emplace_back("bias", nullptr, PluginFieldType::kFLOAT16, 0);
    mFC.nbFields = static_cast<int>(mFields.size());
    mFC.fields = mFields.data();
}

const PluginFieldCollection* AnyPrecisionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* AnyPrecisionPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    int N = 0, K = 0, seed = 0, parent = 0, defPrec = 0;
    std::vector<int> supported;
    const int32_t* qweight = nullptr;
    const __half* luts = nullptr;
    const __half* bias = nullptr;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const PluginField& f = fc->fields[i];
        const std::string key(f.name);
        if (key == "N") N = *static_cast<const int*>(f.data);
        else if (key == "K") K = *static_cast<const int*>(f.data);
        else if (key == "seed_bits") seed = *static_cast<const int*>(f.data);
        else if (key == "parent_bits") parent = *static_cast<const int*>(f.data);
        else if (key == "default_precision") defPrec = *static_cast<const int*>(f.data);
        else if (key == "supported_bits")
        {
            const int* d = static_cast<const int*>(f.data);
            supported.assign(d, d + f.length);
        }
        else if (key == "qweight") qweight = static_cast<const int32_t*>(f.data);
        else if (key == "luts") luts = static_cast<const __half*>(f.data);
        else if (key == "bias" && f.length > 0) bias = static_cast<const __half*>(f.data);
    }

    auto* plugin = new AnyPrecisionPlugin(N, K, seed, parent, supported, defPrec,
                                          qweight, luts, bias);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2* AnyPrecisionPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    auto* plugin = new AnyPrecisionPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

} // namespace anyprec

// Auto-register the creator with the default ("") namespace when the .so loads.
REGISTER_TENSORRT_PLUGIN(anyprec::AnyPrecisionPluginCreator);

// ---- C ABI for the Python runtime (ctypes) ---------------------------------
extern "C" {

// Set/get the active precision read by every plugin instance in enqueue().
void ap_set_precision(int bits) { anyprec::setCurrentPrecision(bits); }
int ap_get_precision() { return anyprec::getCurrentPrecision(); }

// Idempotent explicit registration (in case REGISTER_TENSORRT_PLUGIN was
// stripped by the linker for a statically-linked build).
bool ap_register_plugin()
{
    static anyprec::AnyPrecisionPluginCreator creator;
    auto* registry = getPluginRegistry();
    if (registry == nullptr)
        return false;
    return registry->registerCreator(creator, "");
}

} // extern "C"
