#ifndef InferenceRuntimeConfig_H
#define InferenceRuntimeConfig_H

#ifdef USE_PYTORCH

#include "IOdictionary.H"
#include "Switch.H"
#include "fileName.H"
#include "label.H"
#include "word.H"
#include "InferenceBackend.H"

namespace Foam
{

// Parsed runtime settings for the USE_PYTORCH inference path. This keeps the
// case-driven backend selection and validation logic out of dfChemistryModel.C
// so the chemistry model can focus on orchestration instead of config details.
struct InferenceRuntimeConfig
{
    Switch torchEnabled;
    Switch gpuEnabled;
    Switch verbose;
    label coresPerNode;
    word backendType;
    fileName backendModule;
    fileName artifactPath;
    word executionProvider;
    label intraOpThreads;
    label inputFeatureSize;
    label outputFeatureSize;
    label deviceId;
    word inputTensorName;
    word outputTensorName;
    Switch usePinnedHostMemory;
    Switch centralized;
    bool useThermoTranNN;

    InferenceRuntimeConfig()
    :
        torchEnabled(false),
        gpuEnabled(false),
        verbose(false),
        coresPerNode(8),
        backendType("pytorchEmbedded"),
        backendModule("inference"),
        artifactPath(""),
        executionProvider("cpu"),
        intraOpThreads(1),
        inputFeatureSize(0),
        outputFeatureSize(0),
        deviceId(0),
        inputTensorName(""),
        outputTensorName(""),
        usePinnedHostMemory(true),
        centralized(true),
        useThermoTranNN(false)
    {}
};

InferenceRuntimeConfig parseInferenceRuntimeConfig(const IOdictionary& properties);
void validateInferenceRuntimeConfig(const InferenceRuntimeConfig& config);
bool shouldCreateInferenceBackendNow(
    const InferenceRuntimeConfig& config,
    const bool ownsInferenceWork);
InferenceBackendConfig makeInferenceBackendConfig(const InferenceRuntimeConfig& config);

} // End namespace Foam

#endif

#endif
