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
    constexpr std::size_t base = 1024;
    return ((base + batch_size - 1) / batch_size) * batch_size;
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
        void generate_random_data(const std::size_t num_points)
        {
            input_data.x.resize(num_points);
            input_data.y.resize(num_points);
            T x0 = static_cast<T>(-10.0);
            T xf = static_cast<T>(10.0);
            T step_size = (xf - x0) / static_cast<T>(num_points - 1);
            for(std::size_t i = 0; i < num_points; i++)
            {
                input_data.x[i] = x0 + i * step_size;
                input_data.y[i] = function(input_data.x[i]);
                input_data.y[i] += dist<T>(rng) * noise_level * std::fabs(input_data.y[i]);
            }
        }
        void load_data_from_file(const std::string &file_path)
        {
            input_data.x.clear();
            input_data.y.clear();
            std::ifstream infile(file_path, std::ios_base::in);
            if(!infile.is_open())
            {
                throw std::runtime_error("Failed to open input file: " + file_path);
            }
            T x, y;
            std::string line;
            while(std::getline(infile, line))
            {
                std::istringstream line_stream(line);
                if(line_stream >> x >> y)
                {
                    input_data.x.push_back(x);
                    input_data.y.push_back(y);
                }
            }
        }
    public:
        InputProvider() = default;
        InputProvider(const std::optional<std::string> &input_file_path, const std::size_t batch_size)
        {
            this->batch_size = batch_size;
            if(input_file_path.has_value())
            {
                load_data_from_file(input_file_path.value());
            }
            else 
            {
                generate_random_data(num_random_points(batch_size));
            }
        }
        ~InputProvider() = default;
        const Data<T>& get_data() const { return input_data;};
        const std::vector<T>& get_x() const { return input_data.x;}
        const std::vector<T>& get_y() const { return input_data.y;}
        void set_noise_level(const T new_noise_level) { noise_level = new_noise_level; }
        /**
         * Very technically, this is a sliding window not a batch
         * However, if you loop through data with index incriments equal to batch_size, it will function as a batch provider
         */
        BatchedView<T> get_batch(std::size_t start_index) const
        {
            const std::size_t total = input_data.size();
            if(start_index >= total)
            {
                return { {}, {} };
            }
            std::size_t sz = std::min(batch_size, (total - start_index));
            return {
                std::span<const T>(input_data.x.data() + start_index, sz),
                std::span<const T>(input_data.y.data() + start_index, sz)
            };
        }
};


#endif 