#include "onnx_model.h"
#include <iostream>
template class ONNXModel<float>;

template<Arithmetic T>
void ONNXModel<T>::init(const std::string &model_path, const int64_t batch_size) 
{
    this->batch_size = batch_size;
    session = std::make_unique<Ort::Session>(getONNXEnv(), model_path.c_str(), getSessionOptions());
    input_count = session->GetInputCount();
    output_count = session->GetOutputCount();
    input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    input_shape[0] = batch_size;
    for(auto &dim : input_shape)    
    {
        if(dim < 0) dim = 1;
        input_size *= dim;
    }
    input_strs = session->GetInputNames();
    output_strs = session->GetOutputNames();
    input_names.reserve(input_count);
    output_names.reserve(output_count);
    for(size_t i = 0; i < input_count; i++)
    {
        input_names.push_back(input_strs[i].c_str());
    }
    for(size_t i = 0; i < output_count; i++)
    {
        output_names.push_back(output_strs[i].c_str());
    }
    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_data.resize(input_size);
    input_tensor = Ort::Value::CreateTensor<T>(memory_info, input_data.data(), input_size, input_shape.data(), input_shape.size());
    std::memset(input_tensor.GetTensorMutableData<T>(), 0, input_size * sizeof(T));
}

template<Arithmetic T>
void ONNXModel<T>::reset_input_tensor()
{
    std::memset(input_tensor.GetTensorMutableData<T>(), 0, input_size * sizeof(T));
}

template<Arithmetic T>
void ONNXModel<T>::set_input_tensor(std::span<const T> input) 
{
    if(input.size() != input_size)
    {
        std::string error_msg = "Input size " + std::to_string(input.size()) + " does not match model's expected input size " + std::to_string(input_size) + ".";
        throw std::runtime_error(error_msg);
    }
    std::memcpy(input_tensor.GetTensorMutableData<T>(), input.data(), input_size * sizeof(T));
}

template<Arithmetic T>
std::vector<T> ONNXModel<T>::inference(std::span<const T> input)
{
    set_input_tensor(input);
    std::vector<Ort::Value> outputs = session->Run(run_options, input_names.data(), &input_tensor, 1, output_names.data(), output_count);
    const T* output_ptr = outputs[0].GetTensorData<T>();
    std::size_t output_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return std::vector<T>(output_ptr, output_ptr + output_size);
}