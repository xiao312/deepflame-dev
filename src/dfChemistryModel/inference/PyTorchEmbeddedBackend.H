#ifndef PyTorchEmbeddedBackend_H
#define PyTorchEmbeddedBackend_H

#ifdef USE_PYTORCH

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <string>
#include <vector>

namespace Foam
{

class PyTorchEmbeddedBackend
{
private:
    bool initialized_;
    std::string moduleName_;
    pybind11::module_ module_;

    void ensureInitialized();

public:
    explicit PyTorchEmbeddedBackend(const std::string& moduleName = "inference");

    std::vector<double> inferFlat(
        const std::vector<double>& flatInput,
        std::size_t outputSize);
};

} // End namespace Foam

#endif

#endif
