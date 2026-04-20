#ifndef DNNInferencer_H
#define DNNInferencer_H

#include <torch/script.h>

#include <memory>
#include <vector>

class DNNInferencer
{
public:
    DNNInferencer();
    explicit DNNInferencer(torch::jit::script::Module torchModel);
    DNNInferencer(
        torch::jit::script::Module torchModel0,
        torch::jit::script::Module torchModel1,
        torch::jit::script::Module torchModel2,
        std::string device
    );
    ~DNNInferencer();

    DNNInferencer(const DNNInferencer&);
    DNNInferencer& operator=(const DNNInferencer&);
    DNNInferencer(DNNInferencer&&) noexcept;
    DNNInferencer& operator=(DNNInferencer&&) noexcept;

    // Legacy single-model API kept for compatibility.
    at::Tensor Inference(torch::Tensor inputs);

    // Multi-model API kept for compatibility with dfChemistryModel.
    std::vector<std::vector<double>> Inference_multiDNNs(
        const std::vector<std::vector<double>>& DNNinputs,
        int dimension
    );

private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

#endif
