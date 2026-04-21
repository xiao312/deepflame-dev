#ifndef TensorRtBackend_H
#define TensorRtBackend_H

#ifdef USE_PYTORCH

#include "InferenceBackend.H"

#ifdef USE_TENSORRT
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#endif

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace Foam
{

class TensorRtBackend
:
    public InferenceBackend
{
private:
    fileName enginePath_;
    int configuredInputFeatureSize_;
    int configuredOutputFeatureSize_;
    int deviceId_;
    word configuredInputTensorName_;
    word configuredOutputTensorName_;
    bool usePinnedHostMemory_;

    std::size_t inputFeatureSize_;
    std::size_t outputFeatureSize_;
    std::size_t batchCapacity_;

    std::string inputName_;
    std::string outputName_;

#ifdef USE_TENSORRT
    template<class T>
    struct TensorRtDeleter
    {
        void operator()(T* ptr) const
        {
            if (!ptr)
            {
                return;
            }
#if NV_TENSORRT_MAJOR >= 10
            delete ptr;
#else
            ptr->destroy();
#endif
        }
    };

    class Logger;

    std::unique_ptr<nvinfer1::IRuntime, TensorRtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TensorRtDeleter<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TensorRtDeleter<nvinfer1::IExecutionContext>> context_;

    cudaStream_t stream_;
    float* hostInputBuffer_;
    float* hostOutputBuffer_;
    float* deviceInputBuffer_;
    float* deviceOutputBuffer_;

    static Logger& logger();
    void ensureCapacity(std::size_t batchSize);
    void freeBuffers();
#endif

    void ensureInitialized();

public:
    explicit TensorRtBackend(const InferenceBackendConfig& config);
    ~TensorRtBackend() override;

    std::vector<double> inferFlat(
        const std::vector<double>& flatInput,
        std::size_t outputSize) override;
};

} // End namespace Foam

#endif

#endif
