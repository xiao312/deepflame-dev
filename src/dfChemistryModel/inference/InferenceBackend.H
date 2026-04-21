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

// Runtime configuration shared by all inference backends. The solver owns the
// CFD-side orchestration and routes flat per-cell tensors through one backend
// instance per participating process.
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
    bool verbose;

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
        usePinnedHostMemory(true),
        verbose(false)
    {}
};

class InferenceBackend
{
public:
    virtual ~InferenceBackend() = default;

    static autoPtr<InferenceBackend> New(const InferenceBackendConfig& config);

    // Execute one batched inference call.
    // flatInput stores a contiguous [numCells, numFeatures] tensor flattened in
    // row-major order. outputSize is the total number of doubles expected by the
    // CFD caller after backend execution.
    virtual std::vector<double> inferFlat(
        const std::vector<double>& flatInput,
        std::size_t outputSize) = 0;
};

} // End namespace Foam

#endif

#endif
