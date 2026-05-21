#include "CLI/CLI.hpp"
#include "onnx_model.h"
#include "torch_model.h"
#include "main.h"
#include "data_provider.h"
#include "benchmarker.h"

/*
TODO: move CMake, add comparison run mode that needs 2 -b 
to compare b/n, add option for outputting results to csv or .dat,
then load gnu submodule again
One-liner for now: 
cmake .. -DONNXRUNTIME_DIR=/home/ksalamone59/src/muse/external/onnxruntime/ -DLIBTORCH_DIR=/home/ksalamone59/Packages/libtorch/ && make
*/

int main(int argc, char** argv)
{
    CLI::App app{"Inference Profiler"};

    std::unique_ptr<IBackendBase<float>> backend;
    std::string model_path;
    BackendType backendType;
    std::size_t batch_size;

    RunMode runMode;
    app.add_option("-b,--backend", backendType, "Backend type")->required()->transform(
        CLI::CheckedTransformer(std::map<std::string, BackendType>{
            {"onnxruntime", BackendType::ONNXRunTime},
            {"onnx", BackendType::ONNXRunTime},
            {"ort", BackendType::ONNXRunTime},
            {"torch", BackendType::Torch},
            {"pytorch", BackendType::Torch}
        })
    );
    app.add_option("-r,--run-mode", runMode, "Run mode")->transform(
        CLI::CheckedTransformer(std::map<std::string, RunMode>{
            {"inference", RunMode::Inference},
            {"benchmark", RunMode::Benchmark}
        })
    )->default_val("inference");
    auto num_iterations_opt = app.add_option("-n,--num-iterations", "Number of benchmark iterations. Only useful in benchmark mode.");
    auto input_model_path_opt = app.add_option("-m,--model-path", model_path, "Path to model file if applicable, without file extension.")
    ->default_val("../../model_files/function_model");
    auto input_file_opt = app.add_option("-i,--input-file","Path to input data file if exists. If not given, random data will be generated.");
    app.add_option("--batch-size", batch_size, "Batch size for inference when in batch mode")->default_val(1);
    CLI11_PARSE(app, argc, argv);

    std::optional<std::string> input_file_path;
    switch (input_file_opt->count())
    {
        case 0:
            std::cout << "No input file provided. Random data will be generated for inference." << std::endl;
            break;
        case 1:
            input_file_path = input_file_opt->as<std::string>();
            std::cout << "Input file provided: " << input_file_path.value() << std::endl;
            break;
        default:
            throw std::runtime_error("Multiple input files provided. Only one is supported.");
    }
    switch(input_model_path_opt->count())
    {
        case 0:
            std::cout << "No model path provided. Using default: " << model_path << std::endl;
            break;
        case 1:
            std::cout << "Model path provided: " << model_path << std::endl;
            break;
        default:
            throw std::runtime_error("Multiple model paths provided. Only one is supported.");
    }
    std::unique_ptr<InputProvider<float>> input_provider = std::make_unique<InputProvider<float>>(input_file_path, batch_size);
    switch (backendType)
    {
        case BackendType::ONNXRunTime:
            // TODO: move CMake to top level, make top level build, code easier path
            backend = std::make_unique<ONNXModel<float>>(model_path + ".onnx", batch_size);
            break;
        case BackendType::Torch:
            backend = std::make_unique<TorchModel<float>>(model_path + ".pt", batch_size);
            break;
        default:
            throw std::runtime_error("Unsupported backend type " + std::to_string(static_cast<int>(backendType)));
    }
    std::cout << "Using backend: " << backend->name() << std::endl;
    std::cout << "Generated " << input_provider->get_data().size() << " data points!\n";
    switch (runMode)
    {
        case RunMode::Inference:
        {
            for(std::size_t i=0;i<input_provider->get_data().size();i+=batch_size)
            {
                auto batch = input_provider->get_batch(i);
                auto output = backend->inference(batch.x);
            }
            break;
        }
        case RunMode::Benchmark:
        {
            benchmarker bench(*backend, *input_provider);
            if(num_iterations_opt->count() == 1)
            {
                bench.set_num_iterations(num_iterations_opt->as<std::size_t>());
            }
            bench.warmup();
            bench.time_inference();
            std::cout << bench << std::endl;
            break;
        }
        default:
            throw std::runtime_error("Unsupported run mode " + std::to_string(static_cast<int>(runMode)));
    }
    std::cout << "Done!\n";
    return 0;
}