#ifndef InferenceBackend_H
#define InferenceBackend_H

#ifdef USE_PYTORCH

#include "autoPtr.H"
#include "fileName.H"
#include "word.H"

#include <cstddef>
#include <vector>

namespace Foam
{

struct InferenceBackendConfig
{
    word backendType;
    fileName moduleName;
    fileName artifactPath;
    word executionProvider;
    int intraOpThreads;
    int inputFeatureSize;
    int outputFeatureSize;
    int deviceId;
    word inputTensorName;
    word outputTensorName;
    bool usePinnedHostMemory;

    InferenceBackendConfig()
    :
        backendType("pytorchEmbedded"),
        moduleName("inference"),
        artifactPath(""),
        executionProvider("cpu"),
        intraOpThreads(1),
        inputFeatureSize(0),
        outputFeatureSize(0),
        deviceId(0),
        inputTensorName(""),
        outputTensorName(""),
        usePinnedHostMemory(true)
    {}
};

class InferenceBackend
{
public:
    virtual ~InferenceBackend() = default;

    static autoPtr<InferenceBackend> New(const InferenceBackendConfig& config);

    virtual std::vector<double> inferFlat(
        const std::vector<double>& flatInput,
        std::size_t outputSize) = 0;
};

} // End namespace Foam

#endif

#endif
