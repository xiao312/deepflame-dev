#include "PyTorchEmbeddedBackend.H"

#ifdef USE_PYTORCH

#include "error.H"

#include <chrono>

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
        const auto start = std::chrono::steady_clock::now();
        module_ = pybind11::module_::import(moduleName_.c_str());
        initialized_ = true;
        const auto stop = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();
        Foam::Info << "[PYBIND TIMING] module import module=" << moduleName_
                   << " seconds=" << dt << Foam::nl << Foam::endl;
    }
}

std::vector<double> PyTorchEmbeddedBackend::inferFlat(
    const std::vector<double>& flatInput,
    std::size_t outputSize)
{
    static label callCount = 0;
    static double cumulativeArrayTime = 0.0;
    static double cumulativePythonCallTime = 0.0;
    static double cumulativeCopyTime = 0.0;
    static double cumulativeTotalTime = 0.0;

    ensureInitialized();

    const auto startTotal = std::chrono::steady_clock::now();

    const auto startArray = std::chrono::steady_clock::now();
    auto input = pybind11::array_t<double>(
        {flatInput.size()},
        {sizeof(double)},
        flatInput.data());
    const auto stopArray = std::chrono::steady_clock::now();

    const auto startPythonCall = std::chrono::steady_clock::now();
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>
        result = module_.attr("inference")(input);
    const auto stopPythonCall = std::chrono::steady_clock::now();

    if (result.size() < static_cast<pybind11::ssize_t>(outputSize))
    {
        FatalErrorInFunction
            << "PyTorch backend returned fewer values than expected. expected="
            << outputSize << ", got=" << result.size()
            << abort(FatalError);
    }

    const auto startCopy = std::chrono::steady_clock::now();
    std::vector<double> output(result.data(), result.data() + outputSize);
    const auto stopCopy = std::chrono::steady_clock::now();
    const auto stopTotal = std::chrono::steady_clock::now();

    const double arrayTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopArray - startArray).count();
    const double pythonCallTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopPythonCall - startPythonCall).count();
    const double copyTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopCopy - startCopy).count();
    const double totalTime = std::chrono::duration_cast<std::chrono::duration<double>>(stopTotal - startTotal).count();

    ++callCount;
    cumulativeArrayTime += arrayTime;
    cumulativePythonCallTime += pythonCallTime;
    cumulativeCopyTime += copyTime;
    cumulativeTotalTime += totalTime;

    Foam::Info << "[PYBIND TIMING] call=" << callCount
               << " inputSize=" << flatInput.size()
               << " outputSize=" << outputSize
               << " array=" << arrayTime
               << " pythonCall=" << pythonCallTime
               << " copy=" << copyTime
               << " total=" << totalTime
               << " cumulativeTotal=" << cumulativeTotalTime
               << Foam::nl << Foam::endl;

    return output;
}

} // End namespace Foam

#endif
