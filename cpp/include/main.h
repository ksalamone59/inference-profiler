#ifndef MAIN_H
#define MAIN_H

#include <random>
#include <cmath>

// More to be added 
enum class BackendType 
{
    ONNXRunTime
};

enum class RunMode
{
    Inference,
    Benchmark
};
inline std::mt19937 rng(42); // Fixed seed for reproducibility

inline float function(float x) noexcept 
{
    return std::sin(x) * std::exp(-0.1f * x * x);
}

#endif 