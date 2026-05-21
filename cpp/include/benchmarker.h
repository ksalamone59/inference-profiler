/*
    Based on code from pytorch-onnx-cpp-pipeline, modified for generality
*/

#ifndef BENCHMARKER_H
#define BENCHMARKER_H

#include <chrono>
#include <iostream>
#include "backend_base.h"
#include "custom_concepts.h"
#include "data_provider.h"

template<Arithmetic T>
class benchmarker
{
    public:
        using clock = std::chrono::high_resolution_clock;
    private:
        IBackendBase<T> &model;
        InputProvider<T> &input_provider;
        std::size_t num_iterations{1000};
        std::size_t iteration = 0;
        double mean_ms = 0.0;
        double M2 = 0.0;
    public:
        benchmarker(IBackendBase<T> &m, InputProvider<T> &input) : model(m), input_provider(input) {}
        void set_num_iterations(std::size_t n) {num_iterations = n;}
        void update_stats(double value_ms) // Welford iteration
        {            
            iteration++;
            double delta = value_ms - mean_ms;
            mean_ms += delta / iteration;
            double delta2 = value_ms - mean_ms;
            M2 += delta * delta2;
        }
        void reset()
        {
            iteration = 0;
            mean_ms = 0.0;
            M2 = 0.0;
            model.reset_input_tensor();
        }
        void time_inference()
        {
            reset();
            while (iteration < num_iterations)
            {
                auto start = clock::now();
                for(std::size_t i=0;i<input_provider.get_data().size();i+=model.get_batch_size())
                {
                    auto batch = input_provider.get_batch(i);
                    (void)model.inference(batch.x);
                }
                auto end = clock::now();
                double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
                update_stats(milliseconds);
            }
        }
        void warmup(std::size_t n = 100)
        {
            auto x = input_provider.get_batch(0);
            for (std::size_t i = 0; i < n; i++)
            {
                (void)model.inference(x.x);
            }
        }
        friend std::ostream& operator<< <>(std::ostream& os, const benchmarker<T>& b);
};

template<Arithmetic T>
std::ostream& operator<<(std::ostream& os, const benchmarker<T> &b)
{
    double variance = (b.iteration > 1) ? (b.M2 / (b.iteration - 1)) : 0.0;
    double stddev = std::sqrt(variance);
    const double total_samples = static_cast<double>(b.input_provider.get_data().size());
    const double mean_iteration_ms = b.mean_ms;
    double per_sample_ms = mean_iteration_ms / total_samples;
    double seconds = mean_iteration_ms / 1000.0;
    double throughput = (seconds > 0) ? total_samples / seconds : 0.0;
    
    os << "\n[" << b.model.name() << "]. Batch size: " << b.model.get_batch_size() << ". Number of iterations: " << b.num_iterations << "\n";
    os << "Mean iteration time: " << mean_iteration_ms << " ms\n";
    os << "Latency/sample: " << per_sample_ms << " ms\n";
    os << "Stddev (iteration): " << stddev << " ms\n";
    if(seconds > 0) os << "Throughput: " << throughput << " samples/sec\n";
    else os << "Throughput: N/A (mean iteration time of zero seconds!)\n";
    
    return os;
}

#endif 