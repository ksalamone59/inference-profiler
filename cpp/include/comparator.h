#ifndef COMPARATOR_H
#define COMPARATOR_H

#include "benchmarker.h"
#include "metrics.h"
#include <optional>
#include <iomanip>
#include <numeric>
#include <span>
#include <format>

constexpr auto GREEN  = "\033[32m";
constexpr auto BOLD = "\033[1m";
constexpr auto WHITE = "\033[37m";
constexpr auto CYAN   = "\033[36m";
constexpr auto RESET   = "\033[0m";


struct comparison_results
{
    double r2{0.0}, mse{0.0}, mae{0.0}, max_error{0.0};
    double mean_time{0.0}, throughput{0.0}, stddev{0.0};
};

template<Arithmetic T>
class comparator
{
    private:
        IBackendBase<T> &model1, &model2;
        InputProvider<T> &input_provider;
        benchmarker <T> benchmark1, benchmark2;
        comparison_results results1, results2, model_to_model;
    public:
        comparator() = delete;
        ~comparator() = default;    
        comparator(IBackendBase<T> &m1, IBackendBase<T> &m2, InputProvider<T> &input) 
        : model1(m1), model2(m2), input_provider(input), benchmark1(m1, input), benchmark2(m2, input) {}
        void generate_results(std::optional<std::size_t> num_iterations = std::nullopt)
        {
            if(num_iterations.has_value())
            {
                benchmark1.set_num_iterations(num_iterations.value());
                benchmark2.set_num_iterations(num_iterations.value());
            }
            benchmark1.warmup();
            benchmark2.warmup();
            benchmark1.time_inference();
            benchmark2.time_inference();
            results1.mean_time = benchmark1.get_mean_ms();
            results2.mean_time = benchmark2.get_mean_ms();
            results1.throughput = benchmark1.get_throughput();
            results2.throughput = benchmark2.get_throughput();
            results1.stddev = std::sqrt(benchmark1.get_variance());
            results2.stddev = std::sqrt(benchmark2.get_variance());
            const auto &data = input_provider.get_data();
            std::vector<T> predictions1, predictions2, truth;
            predictions1.reserve(data.size());
            predictions2.reserve(data.size());
            truth.reserve(data.size());
            for(std::size_t i=0;i<data.size();i+=model1.get_batch_size())
            {
                auto batch = input_provider.get_batch(i);
                auto pred1 = model1.inference(batch.x);
                auto pred2 = model2.inference(batch.x);
                predictions1.insert(predictions1.end(), pred1.begin(), pred1.end());
                predictions2.insert(predictions2.end(), pred2.begin(), pred2.end());
                truth.insert(truth.end(), batch.y.begin(), batch.y.end());
            }
            std::span<const T> predictions1_span(predictions1.data(), predictions1.size());
            std::span<const T> predictions2_span(predictions2.data(), predictions2.size());
            std::span<const T> truth_span(truth.data(), truth.size());
            results1.mse = metrics::mse(predictions1_span, truth_span);
            results1.mae = metrics::mae(predictions1_span, truth_span);
            results1.max_error = metrics::max_error(predictions1_span, truth_span);
            results1.r2 = metrics::r2(predictions1_span, truth_span);
            results2.mse = metrics::mse(predictions2_span, truth_span);
            results2.mae = metrics::mae(predictions2_span, truth_span);
            results2.max_error = metrics::max_error(predictions2_span, truth_span);
            results2.r2 = metrics::r2(predictions2_span, truth_span);
            model_to_model.mse = metrics::mse(predictions1_span, predictions2_span);
            model_to_model.mae = metrics::mae(predictions1_span, predictions2_span);
            model_to_model.max_error = metrics::max_error(predictions1_span, predictions2_span);
            model_to_model.r2 = metrics::r2(predictions1_span, predictions2_span);
            // Print table nicely 
            std::cout << "Batch size: " << model1.get_batch_size() << "\n";
            std::cout << CYAN << "================ Inference Performance Comparison ================\n" << RESET;
            std::cout << BOLD << WHITE << std::left << std::setw(20) << "Model " << std::setw(15) << "Mean Time (ms)" << std::setw(15) << "Throughput" << std::setw(15) << "Std Dev" << "\n" << RESET;
            std::cout << GREEN << std::left << std::setw(20) << model1.name() << std::setw(15) << results1.mean_time << std::setw(15) << results1.throughput << std::setw(15) << results1.stddev << "\n" << RESET;
            std::cout << GREEN << std::left << std::setw(20) << model2.name() << std::setw(15) << results2.mean_time << std::setw(15) << results2.throughput << std::setw(15) << results2.stddev << "\n" << RESET;
            std::cout << CYAN << "================ Correctness Comparison ================\n" << RESET;
            std::cout << BOLD << WHITE << std::left << std::setw(20) << "Model " << std::setw(15) << "MSE" << std::setw(15) << "MAE" << std::setw(15) << "Max Error" << std::setw(15) << " R^{2}" << "\n" << RESET;
            std::cout << GREEN << std::left << std::setw(20) << model1.name() << std::setw(15) << results1.mse << std::setw(15) << results1.mae << std::setw(15) << results1.max_error<< std::setw(15) << results1.r2<< "\n" << RESET;
            std::cout<< GREEN << std::left<< std::setw(20)<< model2.name()<< std::setw(15)<< results2.mse<< std::setw(15)<< results2.mae<< std::setw(15)<< results2.max_error<< std::setw(15)<< results2.r2<< "\n" << RESET;
            std::cout << CYAN << "================ Model-to-Model Comparison ================\n" << RESET;
            std::cout << BOLD << WHITE << std::left << std::setw(20) << "Model" << std::setw(15) << "MSE" << std::setw(15) << "MAE" << std::setw(15) << "Max Error" << std::setw(15) << "R^2" << "\n" << RESET;
            auto pair_name = std::format("{} - {}", model1.name(), model2.name());
            std::cout << GREEN <<  std::left << std::setw(20) << pair_name << std::setw(15) << model_to_model.mse << std::setw(15) << model_to_model.mae << std::setw(15) << model_to_model.max_error << std::setw(15) << model_to_model.r2 << "\n" << RESET;
        }
};

#endif