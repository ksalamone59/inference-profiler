#include "CLI/CLI.hpp"
#include "onnx_model.h"
#include "main.h"

int main(int argc, char** argv)
{
    CLI::App app{"Inference Profiler"};

    std::unique_ptr<IBackendBase<float>> backend;
    std::string model_path;
    BackendType backendType;
    RunMode runMode;
    app.add_option("-b,--backend", backendType, "Backend type")->required()->transform(
        CLI::CheckedTransformer(std::map<std::string, BackendType>{
            {"onnxruntime", BackendType::ONNXRunTime},
            {"onnx", BackendType::ONNXRunTime},
            {"ort", BackendType::ONNXRunTime}
        })
    );
    app.add_option("-r,--run-mode", runMode, "Run mode")->transform(
        CLI::CheckedTransformer(std::map<std::string, RunMode>{
            {"inference", RunMode::Inference},
            {"benchmark", RunMode::Benchmark}
        })
    )->default_val("inference");
    CLI11_PARSE(app, argc, argv);
    switch (backendType)
    {
        case BackendType::ONNXRunTime:
            model_path = "onnx_file/function_model.onnx";
            backend = std::make_unique<ONNXModel<float>>(model_path);
            break;
        default:
            throw std::runtime_error("Unsupported backend type " + std::to_string(static_cast<int>(backendType)));
    }
    switch (runMode)
    {
        case RunMode::Inference:
            break;
        default:
            throw std::runtime_error("Unsupported run mode " + std::to_string(static_cast<int>(runMode)));
    }
    return 0;
}