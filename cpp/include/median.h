#ifndef MEDIAN_H
#define MEDIAN_H

#include <algorithm>
#include <vector>
#include <cmath>

#if (__cplusplus >= 202002L) 
#include <concepts>
template<std::random_access_iterator Iter>
#else 
template <typename Iter>
#endif 
inline auto median(Iter begin, Iter end, bool absolute = false)
{
    using T = typename std::iterator_traits<Iter>::value_type;
    if(begin == end) return T{0};
    const auto n = std::distance(begin,end);
    const auto mid = n/2;
    std::vector<T> copy(begin, end);
    if(absolute) for(auto &i : copy) i = std::fabs(i);
    std::nth_element(copy.begin(), copy.begin() + mid, copy.end());
    if(n % 2 == 1) return copy[mid];
    auto second_mid = std::max_element(copy.begin(), copy.begin() + mid);
    return (copy[mid] + *second_mid) / static_cast<T>(2);
}

#endif 