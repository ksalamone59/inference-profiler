#ifndef MAIN_H
#define MAIN_H

#include "CLI/CLI.hpp"
#include "data_provider.h"
#include "benchmarker.h"
#include "onnx_model.h"
#include "torch_model.h"
#include "comparator.h"
#include "median.h"
#include <set>
#include <map>

// More to be added 
enum class BackendType 
{
    ONNXRunTime,
    Torch
};

inline const auto backend_transformer = CLI::CheckedTransformer(
    std::map<std::string, BackendType>{
        {"onnxruntime", BackendType::ONNXRunTime},
        {"onnx",        BackendType::ONNXRunTime},
        {"ort",         BackendType::ONNXRunTime},
        {"torch",       BackendType::Torch},
        {"pytorch",     BackendType::Torch}
    }
);

template<Arithmetic T>
void set_backend_type(std::unique_ptr<IBackendBase<T>> &backend, BackendType backendType, const std::string &model_path, const int64_t batch_size, CLI::Option *num_threads_opt)
{
    std::size_t num_threads = num_threads_opt->as<std::size_t>();
    switch (backendType)
    {
        case BackendType::ONNXRunTime:
            if(!ort_helpers::onnx_threads_set)
            {
                ort_helpers::set_num_onnx_threads(num_threads);
            }
            backend = std::make_unique<ONNXModel<T>>(model_path + ".onnx", batch_size);
            break;
        case BackendType::Torch:
            if(!torch_helpers::torch_threads_set)
            {
                torch_helpers::set_num_torch_threads(num_threads);
            }
            backend = std::make_unique<TorchModel<T>>(model_path + ".pt", batch_size);
            break;
        default:
            throw std::runtime_error("Unsupported backend type " + std::to_string(static_cast<int>(backendType)));
    }
}

inline std::ostream& operator<<(std::ostream& os, const BackendType& b) 
{
    switch (b) 
    {
        case BackendType::ONNXRunTime: return os << "onnxruntime";
        case BackendType::Torch: return os << "torch";
        default: return os << "unknown";
    }
}


#endif 