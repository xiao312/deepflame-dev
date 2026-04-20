#include "InferenceBackend.H"

#ifdef USE_PYTORCH

#include "OnnxRuntimeBackend.H"
#include "PyTorchEmbeddedBackend.H"
#include "error.H"

namespace Foam
{

autoPtr<InferenceBackend> InferenceBackend::New(
    const word& backendType,
    const fileName& moduleName,
    const fileName& artifactPath,
    const word& executionProvider,
    int intraOpThreads,
    int inputFeatureSize,
    int deviceId)
{
    if (backendType == "pytorchEmbedded")
    {
        return autoPtr<InferenceBackend>(new PyTorchEmbeddedBackend(moduleName));
    }

    if (backendType == "onnxRuntime")
    {
        return autoPtr<InferenceBackend>
        (
            new OnnxRuntimeBackend(
                artifactPath,
                executionProvider,
                intraOpThreads,
                inputFeatureSize,
                deviceId)
        );
    }

    FatalErrorInFunction
        << "Unsupported inference backend '" << backendType << "'. "
        << "Configure TorchSettings.backend in constant/CanteraTorchProperties. "
        << "Currently supported backends: pytorchEmbedded, onnxRuntime"
        << abort(FatalError);

    return autoPtr<InferenceBackend>(nullptr);
}

} // End namespace Foam

#endif
