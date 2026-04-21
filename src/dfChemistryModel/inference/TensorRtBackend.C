#include "TensorRtBackend.H"

#ifdef USE_PYTORCH

#include "error.H"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <vector>

#ifdef USE_TENSORRT

namespace Foam
{

class TensorRtBackend::Logger
:
    public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            Info << "[TRT] " << msg << Foam::nl << Foam::endl;
        }
    }
};

TensorRtBackend::Logger& TensorRtBackend::logger()
{
    static Logger instance;
    return instance;
}

}

#endif

namespace Foam
{

TensorRtBackend::TensorRtBackend(const InferenceBackendConfig& config)
:
    enginePath_(config.artifactPath),
    configuredInputFeatureSize_(config.inputFeatureSize),
    configuredOutputFeatureSize_(config.outputFeatureSize),
    deviceId_(config.deviceId),
    configuredInputTensorName_(config.inputTensorName),
    configuredOutputTensorName_(config.outputTensorName),
    usePinnedHostMemory_(config.usePinnedHostMemory),
    inputFeatureSize_(0),
    outputFeatureSize_(0),
    batchCapacity_(0),
    inputName_(),
    outputName_()
#ifdef USE_TENSORRT
    ,
    runtime_(nullptr),
    engine_(nullptr),
    context_(nullptr),
    stream_(nullptr),
    hostInputBuffer_(nullptr),
    hostOutputBuffer_(nullptr),
    deviceInputBuffer_(nullptr),
    deviceOutputBuffer_(nullptr)
#endif
{}

TensorRtBackend::~TensorRtBackend()
{
#ifdef USE_TENSORRT
    freeBuffers();
    if (stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
#endif
}

#ifdef USE_TENSORRT

void TensorRtBackend::freeBuffers()
{
    if (deviceInputBuffer_)
    {
        cudaFree(deviceInputBuffer_);
        deviceInputBuffer_ = nullptr;
    }
    if (deviceOutputBuffer_)
    {
        cudaFree(deviceOutputBuffer_);
        deviceOutputBuffer_ = nullptr;
    }
    if (hostInputBuffer_)
    {
        if (usePinnedHostMemory_)
        {
            cudaFreeHost(hostInputBuffer_);
        }
        else
        {
            delete[] hostInputBuffer_;
        }
        hostInputBuffer_ = nullptr;
    }
    if (hostOutputBuffer_)
    {
        if (usePinnedHostMemory_)
        {
            cudaFreeHost(hostOutputBuffer_);
        }
        else
        {
            delete[] hostOutputBuffer_;
        }
        hostOutputBuffer_ = nullptr;
    }
    batchCapacity_ = 0;
}

void TensorRtBackend::ensureCapacity(const std::size_t batchSize)
{
    if (batchSize <= batchCapacity_)
    {
        return;
    }

    freeBuffers();

    const std::size_t inputCount = batchSize * inputFeatureSize_;
    const std::size_t outputCount = batchSize * outputFeatureSize_;

    if (usePinnedHostMemory_)
    {
        cudaError_t err = cudaMallocHost(
            reinterpret_cast<void**>(&hostInputBuffer_),
            inputCount * sizeof(float));
        if (err != cudaSuccess)
        {
            FatalErrorInFunction
                << "cudaMallocHost failed for TensorRT input buffer: "
                << cudaGetErrorString(err)
                << abort(FatalError);
        }

        err = cudaMallocHost(
            reinterpret_cast<void**>(&hostOutputBuffer_),
            outputCount * sizeof(float));
        if (err != cudaSuccess)
        {
            FatalErrorInFunction
                << "cudaMallocHost failed for TensorRT output buffer: "
                << cudaGetErrorString(err)
                << abort(FatalError);
        }
    }
    else
    {
        hostInputBuffer_ = new float[inputCount];
        hostOutputBuffer_ = new float[outputCount];
    }

    cudaError_t err = cudaMalloc(
        reinterpret_cast<void**>(&deviceInputBuffer_),
        inputCount * sizeof(float));
    if (err != cudaSuccess)
    {
        FatalErrorInFunction
            << "cudaMalloc failed for TensorRT device input buffer: "
            << cudaGetErrorString(err)
            << abort(FatalError);
    }

    err = cudaMalloc(
        reinterpret_cast<void**>(&deviceOutputBuffer_),
        outputCount * sizeof(float));
    if (err != cudaSuccess)
    {
        FatalErrorInFunction
            << "cudaMalloc failed for TensorRT device output buffer: "
            << cudaGetErrorString(err)
            << abort(FatalError);
    }

    if (!context_->setTensorAddress(inputName_.c_str(), deviceInputBuffer_))
    {
        FatalErrorInFunction
            << "TensorRT failed to bind input tensor '" << inputName_ << "'"
            << abort(FatalError);
    }

    if (!context_->setTensorAddress(outputName_.c_str(), deviceOutputBuffer_))
    {
        FatalErrorInFunction
            << "TensorRT failed to bind output tensor '" << outputName_ << "'"
            << abort(FatalError);
    }

    batchCapacity_ = batchSize;
}

#endif

void TensorRtBackend::ensureInitialized()
{
#ifndef USE_TENSORRT
    FatalErrorInFunction
        << "TensorRT backend requested, but USE_TENSORRT is not enabled in this build."
        << abort(FatalError);
#else
    if (engine_)
    {
        return;
    }

    if (enginePath_.empty())
    {
        FatalErrorInFunction
            << "TorchSettings.trtEngine must be provided when backend=tensorRt"
            << abort(FatalError);
    }

    cudaError_t cudaStatus = cudaSetDevice(deviceId_);
    if (cudaStatus != cudaSuccess)
    {
        FatalErrorInFunction
            << "cudaSetDevice(" << deviceId_ << ") failed for TensorRT backend: "
            << cudaGetErrorString(cudaStatus)
            << abort(FatalError);
    }

    std::ifstream engineFile(enginePath_.c_str(), std::ios::binary);
    if (!engineFile)
    {
        FatalErrorInFunction
            << "Failed to open TensorRT engine file '" << enginePath_ << "'"
            << abort(FatalError);
    }

    engineFile.seekg(0, std::ios::end);
    const std::streamoff engineBytes = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);

    if (engineBytes <= 0)
    {
        FatalErrorInFunction
            << "TensorRT engine file is empty: '" << enginePath_ << "'"
            << abort(FatalError);
    }

    std::vector<char> serializedEngine(static_cast<std::size_t>(engineBytes));
    engineFile.read(serializedEngine.data(), serializedEngine.size());
    if (!engineFile)
    {
        FatalErrorInFunction
            << "Failed to read TensorRT engine file '" << enginePath_ << "'"
            << abort(FatalError);
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger()));
    if (!runtime_)
    {
        FatalErrorInFunction
            << "createInferRuntime failed for TensorRT backend"
            << abort(FatalError);
    }

    engine_.reset(runtime_->deserializeCudaEngine(
        serializedEngine.data(),
        serializedEngine.size()));
    if (!engine_)
    {
        FatalErrorInFunction
            << "TensorRT failed to deserialize engine '" << enginePath_ << "'"
            << abort(FatalError);
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_)
    {
        FatalErrorInFunction
            << "TensorRT failed to create execution context for engine '" << enginePath_ << "'"
            << abort(FatalError);
    }

    cudaStatus = cudaStreamCreate(&stream_);
    if (cudaStatus != cudaSuccess)
    {
        FatalErrorInFunction
            << "cudaStreamCreate failed for TensorRT backend: "
            << cudaGetErrorString(cudaStatus)
            << abort(FatalError);
    }

    const int ioCount = engine_->getNbIOTensors();
    for (int i = 0; i < ioCount; ++i)
    {
        const char* tensorName = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
        {
            if (configuredInputTensorName_.empty() || configuredInputTensorName_ == tensorName)
            {
                inputName_ = tensorName;
            }
        }
        else if (engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            if (configuredOutputTensorName_.empty() || configuredOutputTensorName_ == tensorName)
            {
                outputName_ = tensorName;
            }
        }
    }

    if (inputName_.empty() || outputName_.empty())
    {
        FatalErrorInFunction
            << "TensorRT backend could not resolve input/output tensor names. "
            << "Configured input='" << configuredInputTensorName_ << "' output='"
            << configuredOutputTensorName_ << "'"
            << abort(FatalError);
    }

    const nvinfer1::Dims inputShape = engine_->getTensorShape(inputName_.c_str());
    const nvinfer1::Dims outputShape = engine_->getTensorShape(outputName_.c_str());

    if (inputShape.nbDims != 2 || outputShape.nbDims != 2)
    {
        FatalErrorInFunction
            << "TensorRT backend currently expects rank-2 input/output tensors. "
            << "Input rank=" << inputShape.nbDims
            << ", output rank=" << outputShape.nbDims
            << abort(FatalError);
    }

    if (configuredInputFeatureSize_ > 0)
    {
        inputFeatureSize_ = static_cast<std::size_t>(configuredInputFeatureSize_);
    }
    else if (inputShape.d[1] > 0)
    {
        inputFeatureSize_ = static_cast<std::size_t>(inputShape.d[1]);
    }
    else
    {
        FatalErrorInFunction
            << "TensorRT backend needs a fixed input feature dimension. "
            << "Set TorchSettings.trtInputFeatureSize when engine metadata does not provide it."
            << abort(FatalError);
    }

    if (configuredOutputFeatureSize_ > 0)
    {
        outputFeatureSize_ = static_cast<std::size_t>(configuredOutputFeatureSize_);
    }
    else if (outputShape.d[1] > 0)
    {
        outputFeatureSize_ = static_cast<std::size_t>(outputShape.d[1]);
    }
    else
    {
        FatalErrorInFunction
            << "TensorRT backend needs a fixed output feature dimension. "
            << "Set TorchSettings.trtOutputFeatureSize when engine metadata does not provide it."
            << abort(FatalError);
    }

    Info << "[TRT DEBUG] initialized"
         << " engine=" << enginePath_
         << " inputTensor=" << inputName_
         << " outputTensor=" << outputName_
         << " inputFeatureSize=" << inputFeatureSize_
         << " outputFeatureSize=" << outputFeatureSize_
         << " deviceId=" << deviceId_
         << " pinnedHostMemory=" << usePinnedHostMemory_
         << Foam::nl << Foam::endl;
#endif
}

std::vector<double> TensorRtBackend::inferFlat(
    const std::vector<double>& flatInput,
    const std::size_t outputSize)
{
    static label callCount = 0;
    static double cumulativeConvertTime = 0.0;
    static double cumulativeH2DTime = 0.0;
    static double cumulativeRunTime = 0.0;
    static double cumulativeD2HTime = 0.0;
    static double cumulativeCopyTime = 0.0;
    static double cumulativeTotalTime = 0.0;

    ensureInitialized();

#ifndef USE_TENSORRT
    (void)flatInput;
    (void)outputSize;
    return std::vector<double>();
#else
    if (flatInput.size() % inputFeatureSize_ != 0)
    {
        FatalErrorInFunction
            << "Flat input size " << flatInput.size()
            << " is not divisible by TensorRT input feature size " << inputFeatureSize_
            << abort(FatalError);
    }

    const auto startTotal = std::chrono::steady_clock::now();
    const std::size_t batchSize = flatInput.size() / inputFeatureSize_;

    if (outputFeatureSize_ * batchSize != outputSize)
    {
        FatalErrorInFunction
            << "TensorRT backend expected output size " << (outputFeatureSize_ * batchSize)
            << " from batchSize=" << batchSize
            << " and outputFeatureSize=" << outputFeatureSize_
            << ", but solver requested " << outputSize
            << abort(FatalError);
    }

    ensureCapacity(batchSize);

    const auto startConvert = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < flatInput.size(); ++i)
    {
        hostInputBuffer_[i] = static_cast<float>(flatInput[i]);
    }
    const auto stopConvert = std::chrono::steady_clock::now();

    nvinfer1::Dims inputDims = engine_->getTensorShape(inputName_.c_str());
    inputDims.d[0] = static_cast<int64_t>(batchSize);
    if (!context_->setInputShape(inputName_.c_str(), inputDims))
    {
        FatalErrorInFunction
            << "TensorRT setInputShape failed for tensor '" << inputName_
            << "' batchSize=" << batchSize
            << abort(FatalError);
    }

    const auto startH2D = std::chrono::steady_clock::now();
    cudaError_t cudaStatus = cudaMemcpyAsync(
        deviceInputBuffer_,
        hostInputBuffer_,
        flatInput.size() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream_);
    if (cudaStatus != cudaSuccess)
    {
        FatalErrorInFunction
            << "TensorRT H2D copy failed: " << cudaGetErrorString(cudaStatus)
            << abort(FatalError);
    }
    const auto stopH2D = std::chrono::steady_clock::now();

    const auto startRun = std::chrono::steady_clock::now();
    if (!context_->enqueueV3(stream_))
    {
        FatalErrorInFunction
            << "TensorRT enqueueV3 failed for engine '" << enginePath_ << "'"
            << abort(FatalError);
    }
    const auto stopRun = std::chrono::steady_clock::now();

    const auto startD2H = std::chrono::steady_clock::now();
    cudaStatus = cudaMemcpyAsync(
        hostOutputBuffer_,
        deviceOutputBuffer_,
        outputSize * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream_);
    if (cudaStatus != cudaSuccess)
    {
        FatalErrorInFunction
            << "TensorRT D2H copy failed: " << cudaGetErrorString(cudaStatus)
            << abort(FatalError);
    }

    cudaStatus = cudaStreamSynchronize(stream_);
    if (cudaStatus != cudaSuccess)
    {
        FatalErrorInFunction
            << "TensorRT stream synchronization failed: " << cudaGetErrorString(cudaStatus)
            << abort(FatalError);
    }
    const auto stopD2H = std::chrono::steady_clock::now();

    const auto startCopy = std::chrono::steady_clock::now();
    std::vector<double> result(outputSize);
    for (std::size_t i = 0; i < outputSize; ++i)
    {
        result[i] = static_cast<double>(hostOutputBuffer_[i]);
    }
    const auto stopCopy = std::chrono::steady_clock::now();
    const auto stopTotal = std::chrono::steady_clock::now();

    const double convertTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopConvert - startConvert).count();
    const double h2dTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopH2D - startH2D).count();
    const double runTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopRun - startRun).count();
    const double d2hTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopD2H - startD2H).count();
    const double copyTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopCopy - startCopy).count();
    const double totalTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopTotal - startTotal).count();

    ++callCount;
    cumulativeConvertTime += convertTime;
    cumulativeH2DTime += h2dTime;
    cumulativeRunTime += runTime;
    cumulativeD2HTime += d2hTime;
    cumulativeCopyTime += copyTime;
    cumulativeTotalTime += totalTime;

    Foam::Info << "[TRT TIMING] call=" << callCount
               << " batchSize=" << batchSize
               << " inputSize=" << flatInput.size()
               << " outputSize=" << outputSize
               << " convert=" << convertTime
               << " h2d=" << h2dTime
               << " run=" << runTime
               << " d2h=" << d2hTime
               << " copy=" << copyTime
               << " total=" << totalTime
               << " cumulativeTotal=" << cumulativeTotalTime
               << Foam::nl << Foam::endl;

    return result;
#endif
}

} // End namespace Foam

#endif
