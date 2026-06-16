// anyPrecisionPlugin.h
//
// TensorRT plugin that runs an Any-Precision (LUT bit-packed) linear layer
// inside a TensorRT / TensorRT-LLM engine.
//
// API: IPluginV2DynamicExt (supported by TensorRT 8/9/10). The newer IPluginV3
// API is recommended for TRT 10+; see README.md for the migration notes. V2 is
// used here because it is the most widely documented and is what TensorRT-LLM's
// own bundled plugins used through the 0.x series.
//
// One plugin instance == one nn.Linear replaced by AnyPrecisionLinear. It bakes
// the quantized weights (all bit-planes), every supported LUT, and optional
// bias into the engine, and selects the active precision at runtime through the
// process-global anyprec::getCurrentPrecision().
#pragma once

#include "anyPrecisionKernels.h"

#include <NvInferPlugin.h>
#include <cublas_v2.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace anyprec
{

constexpr const char* kANYPREC_PLUGIN_NAME = "AnyPrecisionLinear";
constexpr const char* kANYPREC_PLUGIN_VERSION = "1";

class AnyPrecisionPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    // Build-time constructor: takes host pointers to the weights. Buffers are
    // copied into owned host vectors so they survive past the creator call and
    // can be re-serialized into the engine.
    AnyPrecisionPlugin(int N, int K, int seedBits, int parentBits,
                       const std::vector<int>& supportedBits,
                       int defaultPrecision,
                       const int32_t* qweight,          // [parentBits, N, K/32]
                       const __half* luts,              // concat over supportedBits
                       const __half* bias);             // [N] or nullptr

    // Deserialize constructor.
    AnyPrecisionPlugin(const void* data, size_t length);

    AnyPrecisionPlugin() = delete;
    ~AnyPrecisionPlugin() override = default;

    // ---- IPluginV2DynamicExt ----
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
        const nvinfer1::DimsExprs* inputs, int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos,
        const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
        int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // ---- IPluginV2Ext ----
    nvinfer1::DataType getOutputDataType(int index,
        const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // ---- IPluginV2 ----
    const char* getPluginType() const noexcept override { return kANYPREC_PLUGIN_NAME; }
    const char* getPluginVersion() const noexcept override { return kANYPREC_PLUGIN_VERSION; }
    int getNbOutputs() const noexcept override { return 1; }
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* ns) noexcept override { mNamespace = ns; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    size_t lutTotalElems() const;
    int lutOffsetElems(int bits) const;  // element offset of LUT for `bits`

    // metadata
    int mN{0};
    int mK{0};
    int mSeedBits{0};
    int mParentBits{0};
    int mDefaultPrecision{0};
    bool mHasBias{false};
    std::vector<int> mSupportedBits;

    // host-resident copies (kept for (re)serialization)
    std::vector<int32_t> mQweightHost;
    std::vector<__half> mLutHost;
    std::vector<__half> mBiasHost;

    // device-resident copies (allocated in initialize())
    int32_t* mQweightDev{nullptr};
    __half* mLutDev{nullptr};
    __half* mBiasDev{nullptr};
    cublasHandle_t mCublas{nullptr};
    bool mIsOrin{false};

    std::string mNamespace;
};

class AnyPrecisionPluginCreator : public nvinfer1::IPluginCreator
{
public:
    AnyPrecisionPluginCreator();

    const char* getPluginName() const noexcept override { return kANYPREC_PLUGIN_NAME; }
    const char* getPluginVersion() const noexcept override { return kANYPREC_PLUGIN_VERSION; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name,
        const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name,
        const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* ns) noexcept override { mNamespace = ns; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mFields;
    std::string mNamespace;
};

} // namespace anyprec
