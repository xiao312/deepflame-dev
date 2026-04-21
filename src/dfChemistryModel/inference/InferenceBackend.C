#include "InferenceBackend.H"

#ifdef USE_PYTORCH

#include "OnnxRuntimeBackend.H"
#include "PyTorchEmbeddedBackend.H"
#include "TensorRtBackend.H"
#include "error.H"

namespace Foam
{

autoPtr<InferenceBackend> InferenceBackend::New(const InferenceBackendConfig& config)
{
    if (config.backendType == "pytorchEmbedded")
    {
        return autoPtr<InferenceBackend>(new PyTorchEmbeddedBackend(config));
    }

    if (config.backendType == "onnxRuntime")
    {
        return autoPtr<InferenceBackend>(new OnnxRuntimeBackend(config));
    }

    if (config.backendType == "tensorRt")
    {
#ifdef USE_TENSORRT
        return autoPtr<InferenceBackend>(new TensorRtBackend(config));
#else
        FatalErrorInFunction
            << "backend=tensorRt was requested, but DeepFlame was built without TensorRT support. "
            << "Re-run configure.sh with TensorRT available and rebuild."
            << abort(FatalError);
#endif
    }

    FatalErrorInFunction
        << "Unsupported inference backend '" << config.backendType << "'. "
        << "Configure TorchSettings.backend in constant/CanteraTorchProperties. "
        << "Currently supported backends: pytorchEmbedded, onnxRuntime, tensorRt"
        << abort(FatalError);

    return autoPtr<InferenceBackend>(nullptr);
}

} // End namespace Foam

#endif
