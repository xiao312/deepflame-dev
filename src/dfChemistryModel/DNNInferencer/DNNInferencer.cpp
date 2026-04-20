#include "DNNInferencer.H"

#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr double kBoxCoxLambda = 0.1;
constexpr double kInferenceDeltaT = 1.0e-6;

struct BatchState
{
    torch::Tensor rho;
    torch::Tensor y;
    torch::Tensor yBoxCox;
    torch::Tensor normalized;
};

struct ModelStats
{
    torch::Tensor xmu;
    torch::Tensor xstd;
    torch::Tensor ymu;
    torch::Tensor ystd;
};

struct Timings
{
    double preprocess{0.0};
    double inference{0.0};
    double postprocess{0.0};
    double deviceToHost{0.0};
};

using Clock = std::chrono::steady_clock;

inline ModelStats makeStats(
    const std::vector<double>& xmu,
    const std::vector<double>& xstd,
    const std::vector<double>& ymu,
    const std::vector<double>& ystd,
    const torch::Device& device
)
{
    const auto options = torch::TensorOptions().dtype(torch::kFloat).device(device);

    return ModelStats{
        torch::tensor(xmu, options).unsqueeze(0),
        torch::tensor(xstd, options).unsqueeze(0),
        torch::tensor(ymu, options).unsqueeze(0),
        torch::tensor(ystd, options).unsqueeze(0)};
}

inline torch::Tensor speciesIndices(const torch::Tensor& inputs)
{
    return torch::linspace(
               2,
               inputs.sizes()[1] - 2,
               inputs.sizes()[1] - 3,
               inputs.device())
        .toType(torch::kLong);
}

inline torch::Tensor makeBatchTensor(
    const std::vector<double>& flatValues,
    const std::size_t rows,
    const std::int64_t cols,
    const torch::Device& device)
{
    const auto options = torch::TensorOptions().dtype(torch::kFloat);

    if (rows == 0)
    {
        return torch::empty({0, cols}, options.device(device));
    }

    auto cpuTensor = torch::from_blob(
        const_cast<double*>(flatValues.data()),
        {static_cast<long>(rows), cols},
        torch::TensorOptions().dtype(torch::kDouble));

    return cpuTensor.to(torch::kFloat).to(device);
}

inline BatchState preprocessBatch(const torch::Tensor& rawInputs, const ModelStats& stats)
{
    BatchState batch;
    batch.rho = rawInputs.select(1, rawInputs.sizes()[1] - 1).unsqueeze(1);

    const auto yIdx = speciesIndices(rawInputs);
    const auto temperature = rawInputs.select(1, 0).unsqueeze(1);
    const auto pressure = rawInputs.select(1, 1).unsqueeze(1);

    batch.y = torch::index_select(rawInputs, 1, yIdx);
    batch.yBoxCox = (torch::pow(batch.y, kBoxCoxLambda) - 1.0) / kBoxCoxLambda;

    auto features = torch::cat({temperature, pressure, batch.yBoxCox}, 1);
    batch.normalized = (features - stats.xmu) / stats.xstd;
    return batch;
}

inline torch::Tensor runModel(torch::jit::script::Module& model, const torch::Tensor& inputs)
{
    std::vector<torch::jit::IValue> args;
    args.emplace_back(inputs);
    return model.forward(args).toTensor();
}

inline std::vector<double> postprocessBatch(
    const torch::Tensor& modelOutput,
    const BatchState& batch,
    const ModelStats& stats)
{
    if (batch.normalized.size(0) == 0)
    {
        return {};
    }

    const auto yIdx = speciesIndices(batch.normalized);
    auto deltaY = torch::index_select(modelOutput, 1, yIdx);
    deltaY = deltaY * stats.ystd + stats.ymu;

    auto yOut = torch::pow((batch.yBoxCox + deltaY * kInferenceDeltaT) * kBoxCoxLambda + 1.0, 10.0);
    yOut = yOut / torch::sum(yOut, 1, true);
    auto rr = (yOut - batch.y) * batch.rho / kInferenceDeltaT;

    rr = rr.to(torch::kDouble).to(torch::kCPU);
    return std::vector<double>(rr.data_ptr<double>(), rr.data_ptr<double>() + rr.numel());
}

class InferenceBackend
{
public:
    virtual ~InferenceBackend() = default;

    virtual at::Tensor inferSingle(torch::Tensor inputs) = 0;
    virtual std::vector<std::vector<double>> inferMulti(
        const std::vector<std::vector<double>>& inputs,
        int dimension) = 0;
};

class NullBackend : public InferenceBackend
{
public:
    at::Tensor inferSingle(torch::Tensor inputs) override
    {
        throw std::runtime_error("DNNInferencer backend is not initialized");
    }

    std::vector<std::vector<double>> inferMulti(
        const std::vector<std::vector<double>>&,
        int) override
    {
        throw std::runtime_error("DNNInferencer backend is not initialized");
    }
};

class TorchScriptBackend : public InferenceBackend
{
public:
    explicit TorchScriptBackend(torch::jit::script::Module model)
    : device_(torch::kCUDA), singleModel_(std::move(model))
    {
        singleModel_.to(device_);
        singleStats_ = makeStats(
            {1933.118541482812,
             1.2327983023706526,
             -5.705591538151852,
             -6.446971251373195,
             -4.169802387800032,
             -6.1200334699867165,
             -4.266343396329115,
             -2.6007437468608616,
             -0.4049762774428252},
            {716.6568054751183,
             0.43268544913281914,
             2.0857655247141387,
             2.168997234412133,
             2.707064105162402,
             2.2681157746245897,
             2.221785173612795,
             1.5510851480805254,
             0.30283229364455927},
            {175072.98234441387,
             125434.41067566245,
             285397.9376620931,
             172924.8443087139,
             -97451.53428068386,
             -7160.953630852251,
             -9.791262408691773e-10},
            {179830.51132577812,
             256152.83860126554,
             285811.9455262339,
             263600.5448448552,
             98110.53711881173,
             11752.979335965118,
             4.0735353885293555e-09},
            device_);

        std::cout << "load model and parameters successfully" << std::endl;
    }

    TorchScriptBackend(
        torch::jit::script::Module model0,
        torch::jit::script::Module model1,
        torch::jit::script::Module model2,
        std::string device)
    : device_(device), multiModels_{std::move(model0), std::move(model1), std::move(model2)}
    {
        for (auto& model : multiModels_)
        {
            model.to(device_);
        }

        multiStats_ = {
            makeStats(
                {956.4666683951323,
                 1.2621251609602075,
                 -8.482865855078037,
                 -8.60195200775564,
                 -7.5687249938092975,
                 -8.739604352829021,
                 -3.0365348658864555,
                 -4.044646973729736,
                 -0.12868046894653598},
                {144.56082979138094,
                 0.4316114858005481,
                 1.3421800304159297,
                 1.3271564927376922,
                 1.964747648182199,
                 1.1993472911833807,
                 1.2594695379275647,
                 1.3518816605077604,
                 0.17392016053354714},
                {8901.112679962635,
                 27135.624769093312,
                 30141.97503208172,
                 24712.755148584696,
                 -372.9651472886253,
                 -493.34322699725413,
                 -4.31138850114707e-12},
                {8901.112679962635,
                 27135.624769093312,
                 30141.97503208172,
                 24712.755148584696,
                 372.96514728862553,
                 493.3432269972544,
                 9.409165181242247e-11},
                device_),
            makeStats(
                {1933.118541482812,
                 1.2327983023706526,
                 -5.705591538151852,
                 -6.446971251373195,
                 -4.169802387800032,
                 -6.1200334699867165,
                 -4.266343396329115,
                 -2.6007437468608616,
                 -0.4049762774428252},
                {716.6568054751183,
                 0.43268544913281914,
                 2.0857655247141387,
                 2.168997234412133,
                 2.707064105162402,
                 2.2681157746245897,
                 2.221785173612795,
                 1.5510851480805254,
                 0.30283229364455927},
                {175072.98234441387,
                 125434.41067566245,
                 285397.9376620931,
                 172924.8443087139,
                 -97451.53428068386,
                 -7160.953630852251,
                 -9.791262408691773e-10},
                {179830.51132577812,
                 256152.83860126554,
                 285811.9455262339,
                 263600.5448448552,
                 98110.53711881173,
                 11752.979335965118,
                 4.0735353885293555e-09},
                device_),
            makeStats(
                {2717.141719004927,
                 1.2871371577864235,
                 -5.240181052513087,
                 -4.8947914078286345,
                 -3.117070179161789,
                 -4.346362771443917,
                 -4.657258124450032,
                 -4.537442872141596,
                 -0.11656950757756744},
                {141.48030419772115,
                 0.4281422992061657,
                 0.6561518672685264,
                 0.9820405777881894,
                 1.0442969662425572,
                 0.7554583907448359,
                 1.7144519099198097,
                 1.1299391466695952,
                 0.15743252221610685},
                {-611.0636921032669,
                 -915.1244682112174,
                 519.5930550881994,
                 -11.949500174512165,
                 -2660.9187297995336,
                 159.56360614662788,
                 -7.136459430073843e-11},
                {611.0636921032669,
                 915.1244682112174,
                 519.5930550881994,
                 342.3100987934528,
                 2754.8463649064784,
                 313.3717647966624,
                 2.463374792192512e-10},
                device_)};

        std::cout << "load model and parameters successfully" << std::endl;
    }

    at::Tensor inferSingle(torch::Tensor inputs) override
    {
        auto raw = inputs.to(device_);
        auto batch = preprocessBatch(raw, singleStats_);
        auto output = runModel(singleModel_, batch.normalized);
        auto yIdx = speciesIndices(raw);

        auto deltaY = torch::index_select(output, 1, yIdx);
        deltaY = deltaY * singleStats_.ystd + singleStats_.ymu;
        auto yOut = torch::pow((batch.yBoxCox + deltaY * kInferenceDeltaT) * kBoxCoxLambda + 1.0, 10.0);
        yOut = yOut / torch::sum(yOut, 1, true);
        return (yOut - batch.y) * batch.rho / kInferenceDeltaT;
    }

    std::vector<std::vector<double>> inferMulti(
        const std::vector<std::vector<double>>& inputs,
        int dimension) override
    {
        if (inputs.size() != multiModels_.size())
        {
            throw std::runtime_error("expected exactly three model input groups");
        }

        const auto preprocessStart = Clock::now();

        std::array<torch::Tensor, 3> rawBatches;
        std::array<BatchState, 3> batches;
        for (std::size_t i = 0; i < multiModels_.size(); ++i)
        {
            const auto rows = inputs[i].empty() ? 0 : inputs[i].size() / static_cast<std::size_t>(dimension);
            rawBatches[i] = makeBatchTensor(inputs[i], rows, dimension, device_);
            batches[i] = preprocessBatch(rawBatches[i], multiStats_[i]);
        }

        timings_.preprocess += std::chrono::duration<double>(Clock::now() - preprocessStart).count();

        const auto inferenceStart = Clock::now();
        std::array<torch::Tensor, 3> outputs;
        for (std::size_t i = 0; i < multiModels_.size(); ++i)
        {
            outputs[i] = runModel(multiModels_[i], batches[i].normalized);
        }
        timings_.inference += std::chrono::duration<double>(Clock::now() - inferenceStart).count();

        const auto postStart = Clock::now();
        std::vector<std::vector<double>> results;
        results.reserve(multiModels_.size());

        for (std::size_t i = 0; i < multiModels_.size(); ++i)
        {
            const auto deviceToHostStart = Clock::now();
            auto values = postprocessBatch(outputs[i], batches[i], multiStats_[i]);
            timings_.deviceToHost += std::chrono::duration<double>(Clock::now() - deviceToHostStart).count();
            results.emplace_back(std::move(values));
        }

        timings_.postprocess += std::chrono::duration<double>(Clock::now() - postStart).count();
        return results;
    }

private:
    torch::Device device_;
    torch::jit::script::Module singleModel_;
    std::array<torch::jit::script::Module, 3> multiModels_;
    ModelStats singleStats_;
    std::array<ModelStats, 3> multiStats_;
    Timings timings_;
};

} // namespace

class DNNInferencer::Impl
{
public:
    Impl()
    : backend(std::make_shared<NullBackend>())
    {
    }

    explicit Impl(std::shared_ptr<InferenceBackend> backendPtr)
    : backend(std::move(backendPtr))
    {
    }

    std::shared_ptr<InferenceBackend> backend;
};

DNNInferencer::DNNInferencer()
: impl_(std::make_shared<Impl>())
{
}

DNNInferencer::DNNInferencer(torch::jit::script::Module torchModel)
: impl_(std::make_shared<Impl>(std::make_shared<TorchScriptBackend>(std::move(torchModel))))
{
}

DNNInferencer::DNNInferencer(
    torch::jit::script::Module torchModel0,
    torch::jit::script::Module torchModel1,
    torch::jit::script::Module torchModel2,
    std::string device)
: impl_(std::make_shared<Impl>(
      std::make_shared<TorchScriptBackend>(
          std::move(torchModel0),
          std::move(torchModel1),
          std::move(torchModel2),
          std::move(device))))
{
}

DNNInferencer::~DNNInferencer() = default;
DNNInferencer::DNNInferencer(const DNNInferencer&) = default;
DNNInferencer& DNNInferencer::operator=(const DNNInferencer&) = default;
DNNInferencer::DNNInferencer(DNNInferencer&&) noexcept = default;
DNNInferencer& DNNInferencer::operator=(DNNInferencer&&) noexcept = default;

at::Tensor DNNInferencer::Inference(torch::Tensor inputs)
{
    return impl_->backend->inferSingle(std::move(inputs));
}

std::vector<std::vector<double>> DNNInferencer::Inference_multiDNNs(
    const std::vector<std::vector<double>>& DNNinputs,
    int dimension)
{
    return impl_->backend->inferMulti(DNNinputs, dimension);
}
