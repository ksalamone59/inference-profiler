#ifndef ONNX_MODEL_H
#define ONNX_MODEL_H

#include "backend_base.h"
#include "onnxruntime_cxx_api.h"

// Meyers Singleton for env and session options 
inline static Ort::Env& getONNXEnv() noexcept
{
    static Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
    return env;
}
inline static Ort::SessionOptions& getSessionOptions() noexcept 
{
    static Ort::SessionOptions session_options = [](){  
        Ort::SessionOptions options;  
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.EnableMemPattern(); 
        options.EnableCpuMemArena();
        return options;
    }();
    return session_options;
} 


template <Arithmetic T>
class ONNXModel : public IBackendBase<T> 
{
    public:
        // Implement the virtual functions from IBackendBase
        void init(const std::string &model_path) override;
        ONNXModel(const std::string &model_path)
        { 
            init(model_path);
        } 
        ~ONNXModel() override = default;
        std::vector<T> inference_batch(const std::vector<T>& input, const int64_t batch_size) override;
        std::vector<T> inference_sample(const std::vector<T> &input) override;
        std::string name() const override { return "ONNXModel"; }
        // New members 
        inline void reset_input_tensor();
        inline void set_input_tensor(const std::vector<T>& input);
    private:
        std::unique_ptr<Ort::Session> session;
        Ort::ShapeInferContext::Ints input_shape;
        std::vector<std::string> input_strs, output_strs;
        std::vector<const char *> input_names, output_names;
        std::size_t input_count{0}, output_count{0}, input_size{1};
        std::vector<int64_t> batch_shape;
        Ort::Value input_tensor;
        std::vector<T> input_data;
        T *input_ptr{nullptr};
        Ort::MemoryInfo memory_info{nullptr};
        Ort::RunOptions run_options{nullptr};
};

template class ONNXModel<float>;

#endif 