#include "onnx_model.h"

template<typename T>
void ONNXModel<T>::init(const std::string &model_path) 
{
    session = std::make_unique<Ort::Session>(getONNXEnv(), model_path.c_str(), getSessionOptions());
    input_count = session->GetInputCount();
    output_count = session->GetOutputCount();
    input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
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
    input_ptr = input_tensor.GetTensorMutableData<T>();
    std::memset(input_ptr, 0, input_size * sizeof(T));
}

template<typename T>
void ONNXModel<T>::reset_input_tensor()
{
    std::memset(input_ptr, 0, input_size * sizeof(T));
}

template<typename T>
void ONNXModel<T>::set_input_tensor(const std::vector<T> &input) noexcept
{
    std::memcpy(input_ptr, input.data(), input_size * sizeof(T));
}

template<typename T>
std::vector<T> ONNXModel<T>::inference_batch(const std::vector<T> &input, const int64_t batch_size)
{
    auto shape = input_shape;
    shape[0] = batch_size;
    auto batch_tensor = Ort::Value::CreateTensor<T>(
        memory_info,
        const_cast<T*>(input.data()), // ORT Doesn't modify - const cast safe
        input.size(),
        shape.data(),
        shape.size()
    ); 
    std::vector<Ort::Value> outputs = session->Run(run_options, input_names.data(), &batch_tensor, 1, output_names.data(), output_count);
    T* output_ptr = outputs[0].GetTensorData<T>();
    std::size_t output_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return std::vector<T>(output_ptr, output_ptr + output_size);
}

template<typename T>
std::vector<T> ONNXModel<T>::inference_sample(const std::vector<T> &input)
{
    set_input_tensor(input);
    std::vector<Ort::Value> outputs = session->Run(run_options, input_names.data(), &input_tensor, 1, output_names.data(), output_count);
    T* output_ptr = outputs[0].GetTensorData<T>();
    std::size_t output_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    return std::vector<T>(output_ptr, output_ptr + output_size);
}