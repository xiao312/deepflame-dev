#ifndef PyTorchEmbeddedBackend_H
#define PyTorchEmbeddedBackend_H

#ifdef USE_PYTORCH

#include "InferenceBackend.H"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <string>
#include <vector>

namespace Foam
{

class PyTorchEmbeddedBackend
:
    public InferenceBackend
{
private:
    bool initialized_;
    std::string moduleName_;
    pybind11::module_ module_;

    void ensureInitialized();

public:
    explicit PyTorchEmbeddedBackend(const InferenceBackendConfig& config);

    std::vector<double> inferFlat(
        const std::vector<double>& flatInput,
        std::size_t outputSize) override;
};

} // End namespace Foam

#endif

#endif
