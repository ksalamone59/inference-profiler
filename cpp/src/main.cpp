#include "main.h"


int main(int argc, char** argv)
{
    at::set_num_interop_threads(1);
    CLI::App app{"Inference Profiler"};
    app.footer("Run './backend_profiler <subcommand> --help' for subcommand-specific options.");
    std::unique_ptr<IBackendBase<float>> backend;
    std::string model_path;
    BackendType backendType_infer, backendType_bench;
    std::vector<BackendType> backendType_compare;
    std::size_t batch_size;

    auto infer_subcommand = app.add_subcommand("infer", "Run inference on a model. Will require one -b argument");
    auto benchmark_subcommand = app.add_subcommand("benchmark", "Run benchmark on one model. Will require one backend argument");
    auto compare_subcommand = app.add_subcommand("compare", "Compare two models on inference and output performance. Will require two backend arguments");
    auto sweep_subcommand = app.add_subcommand("sweep", "Run benchmarks across multiple batch sizes and output results to a .dat file. Will require one backend argument");

    infer_subcommand->add_option("-b,--backend", backendType_infer, "Backend type")->required()->transform(backend_transformer)->required()->expected(1);
    benchmark_subcommand->add_option("-b,--backend", backendType_bench, "Backend type")->required()->transform(backend_transformer)->required()->expected(1);
    compare_subcommand->add_option("-b,--backend", backendType_compare, "Backend type")->required()->transform(backend_transformer)->required()->expected(2);
    sweep_subcommand->add_option("-b,--backend", backendType_bench, "Backend type")->required()->transform(backend_transformer)->required()->expected(1);

    auto output_dat_opt = app.add_option("-o,--output", "Path to output .dat file for inference or sweep results.")->default_str("results.dat");

    auto num_iterations_opt = app.add_option("-n,--num-iterations", "Number of benchmark iterations. Only useful in benchmark mode.");
    auto input_model_path_opt = app.add_option("-m,--model-path", model_path, "Path to model file if applicable, without file extension.")
    ->default_val("../model_files/function_model");
    auto input_file_opt = app.add_option("-i,--input-file","Path to input data file if exists. If not given, random data will be generated.");
    app.add_option("--batch-size", batch_size, "Batch size for inference.")->default_val(1);
    auto num_threads_opt = app.add_option("-t,--num-threads", "Number of threads to use for inference.")->default_val(1);
    CLI11_PARSE(app, argc, argv);
    if(output_dat_opt->count() > 1)
    {
        throw std::runtime_error("Multiple output paths provided. Only one is supported.");
    }
    if(argc == 1)
    {
        std::cerr << app.help() << std::endl;
        return 0;
    }
    if(num_threads_opt->count() > 1)
    {
        throw std::runtime_error("Multiple values provided for number of threads. Only one is supported.");
    }
    std::cout << "Using " << num_threads_opt->as<std::size_t>() << " threads for inference.\n";
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
    std::cout << "Generated " << input_provider->get_data().size() << " data points!\n";
    if(infer_subcommand->parsed())
    {
        std::cout << "Running in inference mode.\n";
        set_backend_type(backend, backendType_infer, model_path, batch_size, num_threads_opt);
        std::string output_dat_path = output_dat_opt->as<std::string>();
        std::cout << "Output .dat file: " << output_dat_path << std::endl;
        std::ofstream outfile(output_dat_path, std::ios_base::trunc | std::ios_base::out);
        if(!outfile.is_open())
        {
            throw std::runtime_error("Failed to open output file: " + output_dat_opt->as<std::string>());
        }
        outfile << "# x y\n";
        for(std::size_t i=0;i<input_provider->get_data().size();i+=batch_size)
        {
            auto batch = input_provider->get_batch(i);
            auto output = backend->inference(batch.x);
            for(std::size_t j = 0; j < output.size(); j++)
            {
                outfile << batch.x[j] << " " << output[j] << "\n";
            }
        }
        outfile.close();
    }
    else if(benchmark_subcommand->parsed())
    {
        std::cout << "Running in benchmark mode.\n";
        set_backend_type(backend, backendType_bench, model_path, batch_size, num_threads_opt);
        benchmarker bench(*backend, *input_provider);
        if(num_iterations_opt->count() == 1)
        {
            bench.set_num_iterations(num_iterations_opt->as<std::size_t>());
        }
        bench.warmup();
        bench.time_inference();
        std::cout << bench << std::endl;
    }
    else if(compare_subcommand->parsed())
    {
        // We expect 2, but written for generality in case of possible future extensions 
        std::vector<std::unique_ptr<IBackendBase<float>>> backends;
        std::set<BackendType> unique_backends(backendType_compare.begin(), backendType_compare.end());
        if(unique_backends.size() != backendType_compare.size())
        {
            throw std::runtime_error("Duplicate backend types provided for comparison. Please provide only distinct backends.");
        }
        backends.reserve(backendType_compare.size());
        for(const auto &bt : backendType_compare)
        {
            std::unique_ptr<IBackendBase<float>> b;
            set_backend_type(b, bt, model_path, batch_size, num_threads_opt);
            backends.push_back(std::move(b));
        }
        comparator comp(*backends[0], *backends[1], *input_provider);
        comp.generate_results(num_iterations_opt->count() == 1 ? std::optional<std::size_t>(num_iterations_opt->as<std::size_t>()) : std::nullopt);
        comp.print_results();
    }
    else if(sweep_subcommand->parsed())
    {
        std::cout << "Running in sweep mode.\n";
        std::vector<std::size_t> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
        // Ensure the end of file is .dat
        std::string output_dat_path = output_dat_opt->as<std::string>();
        if(!output_dat_path.ends_with(".dat"))
        {
            output_dat_path += ".dat";
        }
        std::ofstream outfile(output_dat_path, std::ios_base::trunc);
        std::cout << "Output .dat file: " << output_dat_path << std::endl;
        if(!outfile.is_open())
        {
            throw std::runtime_error("Failed to open output file: " + output_dat_opt->as<std::string>());
        }
        outfile << "#batch_size\tmean_ms\tstddev_ms\tthroughput_samples_per_sec\tsigma_throughput\n";
        for(const auto &bs : batch_sizes)
        {
            // Given current setup, backend must be reinstantiated for each batch size
            // Possible future improvement to be made
            std::unique_ptr<IBackendBase<float>> b;
            std::unique_ptr<InputProvider<float>> provider = std::make_unique<InputProvider<float>>(input_file_path, bs);
            set_backend_type(b, backendType_bench, model_path, bs, num_threads_opt);
            benchmarker bench(*b, *provider);
            if(num_iterations_opt->count() == 1)
            {
                bench.set_num_iterations(num_iterations_opt->as<std::size_t>());
            }
            for(int i=0;i<10;i++) bench.warmup();
            std::vector<double> times, stddevs;
            times.reserve(10);
            stddevs.reserve(10);
            std::cout << "Warmed up; starting benchmark for batch size " << bs << "...\n";
            for(int i=0;i<10;i++) 
            {
                bench.time_inference();
                times.push_back(bench.get_mean_ms());
                stddevs.push_back(std::sqrt(bench.get_variance()));
            }
            const double mean_ms = median(times.begin(), times.end());
            const double stddev_ms = median(stddevs.begin(), stddevs.end());
            const double throughput = provider->get_data().size() / (mean_ms / 1000.0);
            double sigma_throughput = throughput * (stddev_ms/mean_ms);
            outfile << bs << "\t" << mean_ms << "\t" << stddev_ms << "\t" << throughput << "\t" << sigma_throughput << "\n";
        }
        outfile.close();
    }
    else
    {
        app.failure_message(CLI::FailureMessage::help);
    }
    return 0;
}