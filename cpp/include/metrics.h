#ifndef METRICS_H
#define METRICS_H

#include "custom_concepts.h"
#include <span>
#include <cmath>
#include <stdexcept>

namespace metrics
{
    template<Arithmetic T>
    double mse(std::span<const T> predictions, std::span<const T> truth)
    {
        if(predictions.size() != truth.size())
        {
            throw std::invalid_argument("Predictions and truth spans must have the same size for MSE calculation.");
        }
        double running_sum{0.0};
        for(std::size_t i = 0; i < predictions.size(); i++)
        {
            double diff = static_cast<double>(predictions[i]) - static_cast<double>(truth[i]);
            running_sum += diff * diff;
        }
        return running_sum / static_cast<double>(predictions.size());
    }

    template<Arithmetic T>
    double mae(std::span<const T> predictions, std::span<const T> truth)
    {
        if(predictions.size() != truth.size())
        {
            throw std::invalid_argument("Predictions and truth spans must have the same size for MAE calculation.");
        }
        double running_sum{0.0};
        for(std::size_t i = 0; i < predictions.size(); i++)
        {
            double diff = std::abs(static_cast<double>(predictions[i]) - static_cast<double>(truth[i]));
            running_sum += diff;
        }
        return running_sum / static_cast<double>(predictions.size());
    }

    template<Arithmetic T>
    double max_error(std::span<const T> predictions, std::span<const T> truth)
    {
        if(predictions.size() != truth.size())
        {
            throw std::invalid_argument("Predictions and truth spans must have the same size for Max Error calculation.");
        }
        double max_err{0.0};
        for(std::size_t i = 0; i < predictions.size(); i++)
        {
            double diff = std::fabs(static_cast<double>(predictions[i]) - static_cast<double>(truth[i]));
            if(diff > max_err)
            {
                max_err = diff;
            }
        }
        return max_err;
    }

    template<Arithmetic T>
    double r2(std::span<const T> predictions, std::span<const T> truth)
    {
        if(predictions.size() != truth.size())
        {
            throw std::invalid_argument("Predictions and truth spans must have the same size for R^2 calculation.");
        }
        double mean_truth{0.0};
        for(const auto &val : truth)
        {
            mean_truth += static_cast<double>(val);
        }
        mean_truth /= static_cast<double>(truth.size());

        double ss_tot{0.0}, ss_res{0.0};
        for(std::size_t i = 0; i < predictions.size(); i++)
        {
            double diff_res = static_cast<double>(predictions[i]) - static_cast<double>(truth[i]);
            double diff_tot = static_cast<double>(truth[i]) - mean_truth;
            ss_res += diff_res * diff_res;
            ss_tot += diff_tot * diff_tot;
        }
        return (ss_tot > 0. ? 1.0 - (ss_res / ss_tot) : 1.0);
    }
};

#endif 