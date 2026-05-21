#include "data_provider.h"

template<Arithmetic T>
InputProvider<T>::InputProvider(const std::optional<std::string> &input_file_path, std::size_t batch_size) 
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

template<Arithmetic T>
void InputProvider<T>::generate_random_data(const std::size_t num_points)
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

template<Arithmetic T>
void InputProvider<T>::load_data_from_file(const std::string &file_path)
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

template<Arithmetic T>
BatchedView<T> InputProvider<T>::get_batch(std::size_t i)
{
    const std::size_t total = input_data.size();
    if(i >= total)
    {
        return { {}, {} };
    }
    std::size_t sz = std::min(batch_size, (total - i));
    return {
        std::span<const T>(input_data.x.data() + i, sz),
        std::span<const T>(input_data.y.data() + i, sz)
    };
}

template class InputProvider<float>;