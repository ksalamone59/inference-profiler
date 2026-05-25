#include "torch_model.h"

template class TorchModel<float>;

template<Arithmetic T>
void TorchModel<T>::init(const std::string &model_path, const int64_t batch_size)
{
    this->batch_size = batch_size;
    try
    {
        mod = torch::jit::load(model_path);
        mod.eval();
    }
    catch (const c10::Error& e)
    {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        throw;
    }
    auto params = mod.parameters();
    bool has_params = false;
    for(const auto &p : params)
    {
        if(p.dim() == 2)
        {
            input_features = p.size(1);
            has_params = true;
            break; 
        }
    }
    if(!has_params) throw std::runtime_error("Model has no parameters to infer input shape from.");
    tensor_options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    input_data.resize(batch_size * input_features, static_cast<T>(0));
    input_size = input_data.size();
    input_tensor = torch::from_blob(input_data.data(), {batch_size, input_features}, tensor_options);
    inputs.push_back(input_tensor);
}

template<Arithmetic T>
void TorchModel<T>::reset_input_tensor()
{
    std::memset(input_tensor.data_ptr<T>(), 0, input_size * sizeof(T));
}

template<Arithmetic T>
void TorchModel<T>::set_input_tensor(std::span<const T> input)
{
    if(input.size() != input_size)
    {
        std::string error_msg = "Input size " + std::to_string(input.size()) + " does not match model's expected input size " + std::to_string(input_size) + ".";
        throw std::runtime_error(error_msg);
    }
    std::memcpy(input_tensor.data_ptr<T>(), input.data(), input_size * sizeof(T));
}

template<Arithmetic T>
std::span<const T> TorchModel<T>::inference(std::span<const T> input)
{
    torch::NoGradGuard no_grad; // Ensure no gradients
    set_input_tensor(input);
    output_tensor = mod.forward(inputs).toTensor().contiguous();
    output_data.assign(output_tensor.data_ptr<T>(), output_tensor.data_ptr<T>() + output_tensor.numel());
    return std::span<const T>(output_data);
}