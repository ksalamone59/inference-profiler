#ifndef TORCH_MODEL_H
#define TORCH_MODEL_H

#include "backend_base.h"
#include "torch/script.h"

template<Arithmetic T>
class TorchModel: public IBackendBase<T>
{
    public:
        void init(const std::string &model_path, const int64_t batch_size) override;
        TorchModel(const std::string &model_path, const int64_t batch_size)
        {
            init(model_path, batch_size);
        }
        ~TorchModel() override = default;
        std::vector<T> inference(std::span<const T> input) override;
        std::string_view name() const override { return "Torch"; }
        void reset_input_tensor() override;
        void set_input_tensor(std::span<const T> input) override;
        int64_t get_batch_size() const override {return batch_size;}
    private:
        int64_t batch_size{1}, input_features{1};
        std::size_t input_size{1};
        torch::jit::script::Module mod; 
        torch::TensorOptions tensor_options;
        std::vector<torch::jit::IValue> inputs;
        std::vector<T> input_data;
        torch::Tensor input_tensor;
};

#endif 