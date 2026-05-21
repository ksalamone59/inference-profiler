#ifndef ONNX_MODEL_H
#define ONNX_MODEL_H

#include "backend_base.h"
#include "onnxruntime_cxx_api.h"

// Meyers Singleton for env and session options 
inline Ort::Env& getONNXEnv() noexcept
{
    static Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
    return env;
}

inline Ort::SessionOptions& getSessionOptions() noexcept 
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
        void init(const std::string &model_path, const int64_t batch_size) override;
        ONNXModel(const std::string &model_path, const int64_t batch_size)
        { 
            init(model_path, batch_size);
        } 
        ~ONNXModel() override = default;
        std::vector<T> inference(std::span<const T> input) override;
        std::string_view name() const override { return "ONNXRunTime"; }
        void reset_input_tensor() override;
        void set_input_tensor(std::span<const T> input) override;
        int64_t get_batch_size() const override {return batch_size;}
    private:
        std::unique_ptr<Ort::Session> session;
        Ort::ShapeInferContext::Ints input_shape;
        std::vector<std::string> input_strs, output_strs;
        std::vector<const char *> input_names, output_names;
        std::size_t input_count{0}, output_count{0}, input_size{1};
        Ort::Value input_tensor;
        std::vector<T> input_data;
        int64_t batch_size{1};
        Ort::MemoryInfo memory_info{nullptr};
        Ort::RunOptions run_options{nullptr};
};

#endif 