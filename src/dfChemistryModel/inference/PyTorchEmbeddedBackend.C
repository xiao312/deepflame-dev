#include "PyTorchEmbeddedBackend.H"

#ifdef USE_PYTORCH

#include "error.H"

namespace Foam
{

PyTorchEmbeddedBackend::PyTorchEmbeddedBackend(const std::string& moduleName)
:
    initialized_(false),
    moduleName_(moduleName)
{}

void PyTorchEmbeddedBackend::ensureInitialized()
{
    if (!initialized_)
    {
        module_ = pybind11::module_::import(moduleName_.c_str());
        initialized_ = true;
    }
}

std::vector<double> PyTorchEmbeddedBackend::inferFlat(
    const std::vector<double>& flatInput,
    std::size_t outputSize)
{
    ensureInitialized();

    auto input = pybind11::array_t<double>(
        {flatInput.size()},
        {sizeof(double)},
        flatInput.data());

    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        result = module_.attr("inference")(input);

    if (result.size() < static_cast<pybind11::ssize_t>(outputSize))
    {
        FatalErrorInFunction
            << "PyTorch backend returned fewer values than expected. expected="
            << outputSize << ", got=" << result.size()
            << abort(FatalError);
    }

    return std::vector<double>(result.data(), result.data() + outputSize);
}

} // End namespace Foam

#endif
