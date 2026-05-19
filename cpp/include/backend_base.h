#ifndef BACKEND_BASE_H
#define BACKEND_BASE_H

#include <vector>
#include <string>
#include <concepts>
#include <type_traits>

/* Needed for this class: 
    - Virtual destructor 
    - Initialization (not used heavily by all backends)
    - Inferencing batched data 
    - Also inferencing per sample
    - Name of backend  
*/

template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>; 
template <Arithmetic T>
class IBackendBase
{
    public:
        virtual ~IBackendBase() = default;
        virtual void init(const std::string &model_path) = 0; 
        virtual std::vector<T> inference_batch(const std::vector<T>& input, const int64_t batch_size) = 0;
        virtual std::vector<T> inference_sample(const std::vector<T> &input) = 0;
        virtual std::string name() const = 0;
};

#endif 