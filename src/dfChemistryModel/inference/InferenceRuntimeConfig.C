#include "InferenceRuntimeConfig.H"

#ifdef USE_PYTORCH

namespace Foam
{

InferenceRuntimeConfig parseInferenceRuntimeConfig(const IOdictionary& properties)
{
    InferenceRuntimeConfig config;
    const dictionary& torchSettings = properties.subDict("TorchSettings");

    config.torchEnabled = torchSettings.lookupOrDefault("torch", false);
    config.gpuEnabled = torchSettings.lookupOrDefault("GPU", false);
    config.verbose = torchSettings.lookupOrDefault("log", false);
    config.coresPerNode = torchSettings.lookupOrDefault("coresPerNode", 8);
    config.backendType = torchSettings.lookupOrDefault("backend", word("pytorchEmbedded"));
    config.backendModule = torchSettings.lookupOrDefault("backendModule", fileName("inference"));
    config.useThermoTranNN = properties.lookupOrDefault("useThermoTranNN", false);

    if (config.backendType == "onnxRuntime")
    {
        config.artifactPath = torchSettings.lookupOrDefault("onnxModel", fileName(""));
        config.executionProvider = torchSettings.lookupOrDefault("onnxExecutionProvider", word("cpu"));
        config.intraOpThreads = torchSettings.lookupOrDefault("onnxIntraOpThreads", 1);
        config.inputFeatureSize = torchSettings.lookupOrDefault("onnxInputFeatureSize", 0);
        config.deviceId = torchSettings.lookupOrDefault("onnxDeviceId", 0);
        config.centralized = torchSettings.lookupOrDefault("onnxCentralized", true);
    }
    else if (config.backendType == "tensorRt")
    {
        config.artifactPath = torchSettings.lookupOrDefault("trtEngine", fileName(""));
        config.inputFeatureSize = torchSettings.lookupOrDefault("trtInputFeatureSize", 0);
        config.outputFeatureSize = torchSettings.lookupOrDefault("trtOutputFeatureSize", 0);
        config.deviceId = torchSettings.lookupOrDefault("trtDeviceId", 0);
        config.inputTensorName = torchSettings.lookupOrDefault("trtInputTensorName", word(""));
        config.outputTensorName = torchSettings.lookupOrDefault("trtOutputTensorName", word(""));
        config.usePinnedHostMemory = torchSettings.lookupOrDefault("trtUsePinnedHostMemory", true);
        config.centralized = torchSettings.lookupOrDefault("trtCentralized", true);
        config.gpuEnabled = true;
    }

    return config;
}

void validateInferenceRuntimeConfig(const InferenceRuntimeConfig& config)
{
    if (config.backendType == "pytorchEmbedded")
    {
        if (config.backendModule.empty())
        {
            FatalErrorInFunction
                << "TorchSettings.backendModule must name the Python module used by "
                << "backend=pytorchEmbedded. Example: backendModule \"inference\";"
                << abort(FatalError);
        }
        return;
    }

    if (config.backendType == "onnxRuntime")
    {
        if (config.artifactPath.empty())
        {
            FatalErrorInFunction
                << "TorchSettings.onnxModel must be set when backend=onnxRuntime. "
                << "Example: onnxModel \"h2_full_inference_wrapper.onnx\";"
                << abort(FatalError);
        }
        if (config.executionProvider != "cpu" && config.executionProvider != "cuda")
        {
            FatalErrorInFunction
                << "TorchSettings.onnxExecutionProvider has unsupported value '"
                << config.executionProvider << "'. Valid values: cpu, cuda"
                << abort(FatalError);
        }
        if (config.intraOpThreads < 1)
        {
            FatalErrorInFunction
                << "TorchSettings.onnxIntraOpThreads must be >= 1. Got "
                << config.intraOpThreads
                << abort(FatalError);
        }
        if (config.deviceId < 0)
        {
            FatalErrorInFunction
                << "TorchSettings.onnxDeviceId must be >= 0. Got "
                << config.deviceId
                << abort(FatalError);
        }
        if (config.executionProvider == "cuda" && !config.gpuEnabled)
        {
            FatalErrorInFunction
                << "TorchSettings.GPU must be true when backend=onnxRuntime and "
                << "onnxExecutionProvider=cuda."
                << abort(FatalError);
        }
        return;
    }

    if (config.backendType == "tensorRt")
    {
        if (config.artifactPath.empty())
        {
            FatalErrorInFunction
                << "TorchSettings.trtEngine must be set when backend=tensorRt. "
                << "Example: trtEngine \"h2_full_inference_wrapper_fp32.plan\";"
                << abort(FatalError);
        }
        if (config.inputFeatureSize <= 0)
        {
            FatalErrorInFunction
                << "TorchSettings.trtInputFeatureSize must be > 0 when backend=tensorRt."
                << abort(FatalError);
        }
        if (config.outputFeatureSize <= 0)
        {
            FatalErrorInFunction
                << "TorchSettings.trtOutputFeatureSize must be > 0 when backend=tensorRt."
                << abort(FatalError);
        }
        if (config.inputTensorName.empty() || config.outputTensorName.empty())
        {
            FatalErrorInFunction
                << "TorchSettings.trtInputTensorName and TorchSettings.trtOutputTensorName "
                << "must both be set when backend=tensorRt."
                << abort(FatalError);
        }
        if (config.deviceId < 0)
        {
            FatalErrorInFunction
                << "TorchSettings.trtDeviceId must be >= 0. Got "
                << config.deviceId
                << abort(FatalError);
        }
        if (!config.centralized)
        {
            FatalErrorInFunction
                << "TensorRT backend currently supports only centralized inference. "
                << "Set TorchSettings.trtCentralized true;"
                << abort(FatalError);
        }
#ifndef USE_TENSORRT
        FatalErrorInFunction
            << "TorchSettings.backend=tensorRt was requested, but this DeepFlame build "
            << "does not include TensorRT support. Re-run configure.sh with --tensorrt_dir "
            << "or TENSORRT_ROOT set, then rebuild."
            << abort(FatalError);
#endif
        return;
    }

    FatalErrorInFunction
        << "TorchSettings.backend has unsupported value '" << config.backendType
        << "'. Valid values: pytorchEmbedded, onnxRuntime, tensorRt"
        << abort(FatalError);
}

bool shouldCreateInferenceBackendNow(
    const InferenceRuntimeConfig& config,
    const bool ownsInferenceWork)
{
    return !config.gpuEnabled || config.backendType == "pytorchEmbedded" || ownsInferenceWork;
}

InferenceBackendConfig makeInferenceBackendConfig(const InferenceRuntimeConfig& config)
{
    InferenceBackendConfig backendConfig;
    backendConfig.backendType = config.backendType;
    backendConfig.moduleName = config.backendModule;
    backendConfig.artifactPath = config.artifactPath;
    backendConfig.executionProvider = config.executionProvider;
    backendConfig.intraOpThreads = config.intraOpThreads;
    backendConfig.inputFeatureSize = config.inputFeatureSize;
    backendConfig.outputFeatureSize = config.outputFeatureSize;
    backendConfig.deviceId = config.deviceId;
    backendConfig.inputTensorName = config.inputTensorName;
    backendConfig.outputTensorName = config.outputTensorName;
    backendConfig.usePinnedHostMemory = config.usePinnedHostMemory;
    backendConfig.verbose = config.verbose;
    return backendConfig;
}

} // End namespace Foam

#endif
