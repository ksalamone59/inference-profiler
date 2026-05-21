#ifndef BACKEND_BASE_H
#define BACKEND_BASE_H

#include <string>
#include <vector>
#include <concepts>
#include <span>
#include <type_traits>
#include "custom_concepts.h"

/* Needed for this class: 
    - Virtual destructor 
    - Initialization (not used heavily by all backends)
    - Inferencing batched data 
    - Also inferencing per sample
    - Name of backend  
*/

template <Arithmetic T>
class IBackendBase
{
    public:
        virtual ~IBackendBase() = default;
        virtual void init(const std::string &model_path, const int64_t batch_size) = 0; 
        virtual std::vector<T> inference(std::span<const T> input) = 0;
        virtual std::string_view name() const = 0;
        virtual void reset_input_tensor() = 0;
        virtual void set_input_tensor(std::span<const T> input) = 0;
        virtual int64_t get_batch_size() const = 0;
};

#endif 