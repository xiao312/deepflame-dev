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

class InferenceBackend
{
public:
    virtual ~InferenceBackend() = default;

    static autoPtr<InferenceBackend> New(
        const word& backendType,
        const fileName& moduleName,
        const fileName& artifactPath,
        const word& executionProvider,
        int intraOpThreads,
        int inputFeatureSize,
        int deviceId);

    virtual std::vector<double> inferFlat(
        const std::vector<double>& flatInput,
        std::size_t outputSize) = 0;
};

} // End namespace Foam

#endif

#endif
