#include "OnnxRuntimeBackend.H"

#ifdef USE_PYTORCH

#include "error.H"

#include <algorithm>
#include <chrono>
#include <exception>
#include <sys/stat.h>

namespace
{

Ort::Env& globalOnnxEnv(const int intraOpThreads)
{
    static std::unique_ptr<Ort::ThreadingOptions> threadingOptions;
    static std::unique_ptr<Ort::Env> env;

    if (!env)
    {
        Foam::Info<< "[ONNX DEBUG] creating process-global Ort::ThreadingOptions"
            << " intraOp=" << std::max(intraOpThreads, 1)
            << " interOp=1 spin=0" << Foam::nl << Foam::endl;

        threadingOptions.reset(new Ort::ThreadingOptions());
        threadingOptions->SetGlobalIntraOpNumThreads(std::max(intraOpThreads, 1));
        threadingOptions->SetGlobalInterOpNumThreads(1);
        threadingOptions->SetGlobalSpinControl(0);

        Foam::Info<< "[ONNX DEBUG] creating process-global Ort::Env" << Foam::nl << Foam::endl;
        env.reset(new Ort::Env(*threadingOptions, ORT_LOGGING_LEVEL_WARNING, "DeepFlameOnnxRuntime"));
        Foam::Info<< "[ONNX DEBUG] process-global Ort::Env ready" << Foam::nl << Foam::endl;
    }
    else
    {
        Foam::Info<< "[ONNX DEBUG] reusing process-global Ort::Env" << Foam::nl << Foam::endl;
    }

    return *env;
}

}

namespace Foam
{

OnnxRuntimeBackend::OnnxRuntimeBackend(
    const fileName& modelPath,
    const word& executionProvider,
    int intraOpThreads,
    int configuredInputFeatureSize,
    int deviceId)
:
    modelPath_(modelPath),
    executionProvider_(executionProvider),
    intraOpThreads_(intraOpThreads),
    configuredInputFeatureSize_(configuredInputFeatureSize),
    deviceId_(deviceId),
    session_(nullptr),
    ioBinding_(nullptr),
    inputFeatureSize_(0),
    outputFeatureSize_(0)
{}

Ort::Env& OnnxRuntimeBackend::processGlobalEnv(const int intraOpThreads)
{
    return globalOnnxEnv(intraOpThreads);
}

void OnnxRuntimeBackend::ensureInitialized()
{
    if (session_)
    {
        Foam::Info<< "[ONNX DEBUG] session already initialized for model=" << modelPath_
            << Foam::nl << Foam::endl;
        return;
    }

    if (modelPath_.empty())
    {
        FatalErrorInFunction
            << "TorchSettings.onnxModel must be provided when backend=onnxRuntime"
            << abort(FatalError);
    }

    if (executionProvider_ != "cpu" && executionProvider_ != "cuda")
    {
        FatalErrorInFunction
            << "Unsupported ONNX Runtime execution provider '" << executionProvider_
            << "'. Currently supported: cpu, cuda"
            << abort(FatalError);
    }

    struct stat modelStat;
    const bool modelExists = (stat(modelPath_.c_str(), &modelStat) == 0);
    Foam::Info<< "[ONNX DEBUG] ensureInitialized begin"
        << " model=" << modelPath_
        << " provider=" << executionProvider_
        << " requestedIntraOpThreads=" << intraOpThreads_
        << " configuredInputFeatureSize=" << configuredInputFeatureSize_
        << " deviceId=" << deviceId_
        << " modelExists=" << modelExists;
    if (modelExists)
    {
        Foam::Info<< " modelBytes=" << modelStat.st_size;
    }
    Foam::Info<< Foam::nl << Foam::endl;

    try
    {
        Foam::Info<< "[ONNX DEBUG] configuring Ort::SessionOptions" << Foam::nl << Foam::endl;
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);
        sessionOptions.DisablePerSessionThreads();
        sessionOptions.DisableMemPattern();
        sessionOptions.DisableCpuMemArena();
        sessionOptions.AddConfigEntry("session.intra_op.allow_spinning", "0");
        sessionOptions.AddConfigEntry("session.inter_op.allow_spinning", "0");

        if (executionProvider_ == "cuda")
        {
            Foam::Info<< "[ONNX DEBUG] appending CUDA execution provider deviceId="
                << deviceId_ << Foam::nl << Foam::endl;
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.device_id = deviceId_;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        }

        Foam::Info<< "[ONNX DEBUG] Ort::SessionOptions ready"
            << " graphOpt=ORT_ENABLE_EXTENDED exec=ORT_SEQUENTIAL intra=1 inter=1"
            << " perSessionThreads=off memPattern=off cpuMemArena=off spinning=off"
            << " provider=" << executionProvider_
            << Foam::nl << Foam::endl;

        Foam::Info<< "[ONNX DEBUG] requesting process-global Ort::Env" << Foam::nl << Foam::endl;
        Ort::Env& env = processGlobalEnv(intraOpThreads_);
        Foam::Info<< "[ONNX DEBUG] process-global Ort::Env acquired" << Foam::nl << Foam::endl;

        Foam::Info<< "[ONNX DEBUG] creating Ort::Session for model load/graph optimization" << Foam::nl << Foam::endl;
        session_.reset(new Ort::Session(env, modelPath_.c_str(), sessionOptions));
        Foam::Info<< "[ONNX DEBUG] Ort::Session created successfully" << Foam::nl << Foam::endl;

        if (executionProvider_ == "cuda")
        {
            Foam::Info<< "[ONNX DEBUG] creating Ort::IoBinding for CUDA EP" << Foam::nl << Foam::endl;
            ioBinding_.reset(new Ort::IoBinding(*session_));
            Foam::Info<< "[ONNX DEBUG] Ort::IoBinding ready" << Foam::nl << Foam::endl;
        }

        const std::size_t inputCount = session_->GetInputCount();
        const std::size_t outputCount = session_->GetOutputCount();
        Foam::Info<< "[ONNX DEBUG] session I/O counts inputs=" << inputCount
            << " outputs=" << outputCount << Foam::nl << Foam::endl;

        if (inputCount != 1 || outputCount != 1)
        {
            FatalErrorInFunction
                << "ONNX wrapper model must expose exactly one input and one output. "
                << "Got inputs=" << inputCount
                << ", outputs=" << outputCount
                << abort(FatalError);
        }

        Foam::Info<< "[ONNX DEBUG] creating Ort allocator for metadata queries" << Foam::nl << Foam::endl;
        Ort::AllocatorWithDefaultOptions allocator;
        Foam::Info<< "[ONNX DEBUG] querying input/output names" << Foam::nl << Foam::endl;
        auto inputName = session_->GetInputNameAllocated(0, allocator);
        auto outputName = session_->GetOutputNameAllocated(0, allocator);
        inputName_ = inputName.get();
        outputName_ = outputName.get();
        Foam::Info<< "[ONNX DEBUG] inputName=" << inputName_
            << " outputName=" << outputName_ << Foam::nl << Foam::endl;

        if (configuredInputFeatureSize_ > 0)
        {
            inputFeatureSize_ = static_cast<std::size_t>(configuredInputFeatureSize_);
            Foam::Info
                << "[ONNX DEBUG] skipping input/output shape metadata query; using configured inputFeatureSize="
                << inputFeatureSize_ << Foam::nl << Foam::endl;
        }
        else
        {
            Foam::Info<< "[ONNX DEBUG] querying tensor type/shape metadata" << Foam::nl << Foam::endl;
            auto inputInfo = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            auto outputInfo = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
            auto inputShape = inputInfo.GetShape();
            auto outputShape = outputInfo.GetShape();

            Foam::Info<< "[ONNX DEBUG] input rank=" << inputShape.size()
                << " output rank=" << outputShape.size() << Foam::nl << Foam::endl;

            if (inputShape.size() != 2 || outputShape.size() != 2)
            {
                FatalErrorInFunction
                    << "ONNX wrapper model must use rank-2 input/output tensors. "
                    << "Input rank=" << inputShape.size() << ", output rank=" << outputShape.size()
                    << abort(FatalError);
            }

            if (inputShape[1] <= 0 || outputShape[1] <= 0)
            {
                FatalErrorInFunction
                    << "ONNX wrapper model must have fixed feature dimensions. "
                    << "Input dim1=" << inputShape[1] << ", output dim1=" << outputShape[1]
                    << abort(FatalError);
            }

            inputFeatureSize_ = static_cast<std::size_t>(inputShape[1]);
            outputFeatureSize_ = static_cast<std::size_t>(outputShape[1]);

            Foam::Info<< "[ONNX DEBUG] feature sizes input=" << inputFeatureSize_
                << " output=" << outputFeatureSize_ << Foam::nl << Foam::endl;
        }
        Foam::Info<< "[ONNX DEBUG] ensureInitialized complete" << Foam::nl << Foam::endl;
    }
    catch (const Ort::Exception& e)
    {
        FatalErrorInFunction
            << "ONNX Runtime exception during ensureInitialized for model='" << modelPath_
            << "' provider='" << executionProvider_ << "': " << e.what()
            << abort(FatalError);
    }
    catch (const std::exception& e)
    {
        FatalErrorInFunction
            << "Standard exception during ONNX ensureInitialized for model='" << modelPath_
            << "' provider='" << executionProvider_ << "': " << e.what()
            << abort(FatalError);
    }
}

std::vector<double> OnnxRuntimeBackend::inferFlat(
    const std::vector<double>& flatInput,
    std::size_t outputSize)
{
    static label callCount = 0;
    static double cumulativeConvertTime = 0.0;
    static double cumulativeTensorCreateTime = 0.0;
    static double cumulativeRunTime = 0.0;
    static double cumulativeCopyTime = 0.0;
    static double cumulativeTotalTime = 0.0;

    ensureInitialized();

    if (flatInput.size() % inputFeatureSize_ != 0)
    {
        FatalErrorInFunction
            << "Flat input size " << flatInput.size()
            << " is not divisible by ONNX input feature size " << inputFeatureSize_
            << abort(FatalError);
    }

    const auto startTotal = std::chrono::steady_clock::now();
    const std::size_t batchSize = flatInput.size() / inputFeatureSize_;

    const auto startConvert = std::chrono::steady_clock::now();
    std::vector<float> inputData(flatInput.begin(), flatInput.end());
    std::vector<int64_t> inputDims{static_cast<int64_t>(batchSize), static_cast<int64_t>(inputFeatureSize_)};
    const auto stopConvert = std::chrono::steady_clock::now();

    const auto startTensorCreate = std::chrono::steady_clock::now();
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputData.data(),
        inputData.size(),
        inputDims.data(),
        inputDims.size());
    const auto stopTensorCreate = std::chrono::steady_clock::now();

    const auto startRun = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputTensors;

    if (executionProvider_ == "cuda")
    {
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo outputMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        ioBinding_->ClearBoundInputs();
        ioBinding_->ClearBoundOutputs();
        ioBinding_->BindInput(inputName_.c_str(), inputTensor);
        ioBinding_->BindOutput(outputName_.c_str(), outputMemoryInfo);
        session_->Run(Ort::RunOptions{nullptr}, *ioBinding_);
        outputTensors = ioBinding_->GetOutputValues(allocator);
    }
    else
    {
        const char* inputNames[] = {inputName_.c_str()};
        const char* outputNames[] = {outputName_.c_str()};
        outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);
    }
    const auto stopRun = std::chrono::steady_clock::now();

    if (outputTensors.size() != 1 || !outputTensors[0].IsTensor())
    {
        FatalErrorInFunction
            << "ONNX Runtime did not return a single tensor output"
            << abort(FatalError);
    }

    const auto startCopy = std::chrono::steady_clock::now();
    auto& outputTensor = outputTensors[0];
    auto outputInfo = outputTensor.GetTensorTypeAndShapeInfo();
    const std::size_t valueCount = outputInfo.GetElementCount();
    const float* outputData = outputTensor.GetTensorData<float>();

    if (valueCount < outputSize)
    {
        FatalErrorInFunction
            << "ONNX Runtime backend returned fewer values than expected. expected="
            << outputSize << ", got=" << valueCount
            << abort(FatalError);
    }

    std::vector<double> result(outputSize);
    for (std::size_t i = 0; i < outputSize; ++i)
    {
        result[i] = static_cast<double>(outputData[i]);
    }
    const auto stopCopy = std::chrono::steady_clock::now();
    const auto stopTotal = std::chrono::steady_clock::now();

    const double convertTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopConvert - startConvert).count();
    const double tensorCreateTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopTensorCreate - startTensorCreate).count();
    const double runTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopRun - startRun).count();
    const double copyTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopCopy - startCopy).count();
    const double totalTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopTotal - startTotal).count();

    ++callCount;
    cumulativeConvertTime += convertTime;
    cumulativeTensorCreateTime += tensorCreateTime;
    cumulativeRunTime += runTime;
    cumulativeCopyTime += copyTime;
    cumulativeTotalTime += totalTime;

    Foam::Info << "[ONNX TIMING] call=" << callCount
               << " batchSize=" << batchSize
               << " inputSize=" << flatInput.size()
               << " outputSize=" << outputSize
               << " convert=" << convertTime
               << " tensorCreate=" << tensorCreateTime
               << " run=" << runTime
               << " copy=" << copyTime
               << " total=" << totalTime
               << " cumulativeTotal=" << cumulativeTotalTime
               << Foam::nl << Foam::endl;

    return result;
}

} // End namespace Foam

#endif
