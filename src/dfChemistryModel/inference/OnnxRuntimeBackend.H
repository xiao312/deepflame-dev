#ifndef OnnxRuntimeBackend_H
#define OnnxRuntimeBackend_H

#ifdef USE_PYTORCH

#include "InferenceBackend.H"
#include "third_party/onnxruntime/onnxruntime_cxx_api.h"

#include <memory>
#include <string>
#include <vector>

namespace Foam
{

class OnnxRuntimeBackend
:
    public InferenceBackend
{
private:
    fileName modelPath_;
    word executionProvider_;
    int intraOpThreads_;
    int configuredInputFeatureSize_;
    int deviceId_;

    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::IoBinding> ioBinding_;
    std::size_t inputFeatureSize_;
    std::size_t outputFeatureSize_;
    std::string inputName_;
    std::string outputName_;

    static Ort::Env& processGlobalEnv(int intraOpThreads);
    void ensureInitialized();

public:
    explicit OnnxRuntimeBackend(const InferenceBackendConfig& config);

    std::vector<double> inferFlat(
        const std::vector<double>& flatInput,
        std::size_t outputSize) override;
};

} // End namespace Foam

#endif

#endif
