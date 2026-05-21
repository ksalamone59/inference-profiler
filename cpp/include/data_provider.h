#ifndef DATA_PROVIDER_H
#define DATA_PROVIDER_H

#include <vector>
#include <string>
#include <optional>
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <span>
#include "custom_concepts.h"

inline std::mt19937 rng(42); // Fixed seed for reproducibility
inline std::size_t num_random_points(std::size_t batch_size)
{
    constexpr std::size_t base = 1000;
    return (base / batch_size) * batch_size;
}
template<Arithmetic T>
std::uniform_real_distribution<T> dist(-10.0, 10.0);

template<Arithmetic T>
inline T function(T x) noexcept 
{
    return static_cast<T>(std::sin(x) * std::exp(-0.1 * x * x));
}

template<Arithmetic T>
struct Data 
{
    std::vector<T> x, y;
    std::size_t size() const { return x.size();}
};

template<Arithmetic T>
struct BatchedView
{
    std::span<const T> x, y;
};

template<Arithmetic T>
class InputProvider
{
    private:
        Data<T> input_data;
        T noise_level{0.1};
        std::size_t batch_size{1};
        void generate_random_data(const std::size_t num_points);
        void load_data_from_file(const std::string &file_path);
    public:
        InputProvider() = default;
        InputProvider(const std::optional<std::string> &input_file_path, const std::size_t batch_size);
        ~InputProvider() = default;
        const Data<T>& get_data() const { return input_data;};
        const std::vector<T>& get_x() const { return input_data.x;}
        const std::vector<T>& get_y() const { return input_data.y;}
        void set_noise_level(const T new_noise_level) { noise_level = new_noise_level; }
        BatchedView<T> get_batch(std::size_t i);
};


#endif 