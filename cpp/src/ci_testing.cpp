#include <iostream>
#include <stdexcept>
#include <vector>
#include <span>
#include <ranges>
#include <numeric>
#include "metrics.h"
#include "median.h"
#include "data_provider.h"
#include "custom_concepts.h"

constexpr double EPSILON = 1e-5;

struct test_results
{
    std::string name;
    bool passed;
    std::string message;
};

template<Arithmetic T>
test_results run_test(const T expected, const T actual, const std::string &message, const std::string &name)
{
    bool passed = std::fabs(expected - actual) <= EPSILON;
    return {name, passed, message + ". Expected: " + std::to_string(expected) + ", Actual: " + std::to_string(actual)};
}

static void check_tests(const std::vector<test_results> &output_test_results)
{
    bool any_failures = false;
    int num_pass{0}, num_fail{0};
    for(const auto &result : output_test_results)
    {
        if(!result.passed)
        {
            std::string failure_message = "[FAIL]. Test '" + result.name + "' failed: " + result.message;
            std::cerr << failure_message << std::endl;
            any_failures = true;
            num_fail++;
        }
        else num_pass++;
    }
    std::cout << "Test Summary: " << num_pass + num_fail << " total tests ran.\n";
    std::cout << "\t [PASS] " << num_pass << "\n";
    std::cout << "\t [FAIL] " << num_fail << "\n";
    if(any_failures)
    {
        throw std::runtime_error("One or more tests failed. See above for details.");
    }
}

int main()
{
    try 
    {
        std::vector<test_results> output_test_results;
        std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
        std::vector<double> other_results = {2.0, 4.0, 6.0, 8.0, 10.0};
        std::span<const double> other_span(other_results);
        std::vector<double> results = {2.0, 4.0, 6.0, 8.0, 10.0};
        BatchedView<double> view{std::span<const double>(data), std::span<const double>(results)};
        // Test metrics
        const double r2 = metrics::r2(other_span, view.y);
        const double mse = metrics::mse(view.x, view.y);
        const double mae = metrics::mae(view.x, view.y);
        const double max_diff = metrics::max_error(view.x, view.y);
        const double expected_zero_mse = metrics::mse(other_span, view.y);
        const double expected_zero_mae = metrics::mae(other_span, view.y);
        const double expected_zero_max_diff = metrics::max_error(other_span, view.y);
        output_test_results.push_back(run_test(1.0, r2, "R^2 should be 1.0 for perfectly correlated data.", "R^2 Perfect"));
        output_test_results.push_back(run_test(0.0, expected_zero_mse, "MSE should be 0.0 for perfectly matching data.", "MSE Perfect"));
        output_test_results.push_back(run_test(0.0, expected_zero_mae, "MAE should be 0.0 for perfectly matching data.", "MAE Perfect"));
        output_test_results.push_back(run_test(0.0, expected_zero_max_diff, "Max Error should be 0.0 for perfectly matching data.", "Max Error Perfect"));
        output_test_results.push_back(run_test(11.0, mse, "MSE should be 11.0 for the given data.", "MSE Nonzero"));
        output_test_results.push_back(run_test(3.0, mae, "MAE should be 3.0 for the given data.", "MAE Nonzero"));
        output_test_results.push_back(run_test(5.0, max_diff, "Max Error should be 5.0 for the given data.", "Max Error Nonzero"));
        double non_perfect_r2 = metrics::r2(view.x, view.y);
        output_test_results.push_back(run_test(-0.375, non_perfect_r2, "R^2 should be -0.375 for the given data.", "R^2 Nonzero"));
        const double med = median(data.begin(), data.end());
        output_test_results.push_back(run_test(3.0, med, "Median should be 3.0 for the given data.", "Median, Double Odd"));
        std::vector<int> even_data = {1, 2, 3, 4};
        auto even_data_double_view = even_data | std::views::transform([](int i) { return static_cast<double>(i); });
        const int even_med_int = median(even_data.begin(), even_data.end());
        const double even_med_double = median(even_data_double_view.begin(), even_data_double_view.end());
        output_test_results.push_back(run_test(2, even_med_int, "Median should be 2 for the given even-sized integer data.", "Median, Even Int"));
        output_test_results.push_back(run_test(2.5, even_med_double, "Median should be 2.5 for the given even-sized double data.", "Median, Even Double"));
        // Test data provider:
        InputProvider<double> providerA(std::nullopt, 10);
        InputProvider<double> providerB(std::nullopt, 10);
        output_test_results.push_back(run_test(providerA.get_x()[0], providerB.get_x()[0], "First x value should be the same for both providers with the same seed.", "Random X Value Equivalence"));
        output_test_results.push_back(run_test(providerA.get_y()[0], providerB.get_y()[0], "First y value should be the same for both providers with the same seed.", "Random Y Value Equivalence"));
        auto batch = providerA.get_batch(0);
        auto batch1 = providerA.get_batch(1);
        output_test_results.push_back(run_test(10UL, batch.x.size(), "Batch size should be 10 for the first batch.", "Proper Batch Size"));
        output_test_results.push_back(run_test(batch.x[1], batch1.x[0], "Second batch should start where the first batch left off.", "Batch Value Continuity"));
        output_test_results.push_back(run_test(1030UL, providerA.get_data().size(), "Total data size should be 1030 for batch size of 10.", "Random Data Data Size"));
        auto empty = providerA.get_batch(100000);
        output_test_results.push_back(run_test(0UL, empty.x.size(), "Batch should be empty when requesting beyond data size.", "Empty Batch Size"));
        InputProvider<double> providerC("../test_input.dat", 1);
        output_test_results.push_back(run_test(3UL, providerC.get_data().size(), "Data provider should load 3 points from the test input file.", "File Proper Data Size"));
        output_test_results.push_back(run_test(1.01, providerC.get_x()[0], "First x value should be 1.0 from the test input file.", "File X Value Equivalence"));
        output_test_results.push_back(run_test(9.4, providerC.get_y().back(), "Last y value should be 9.4 from the test input file.", "File Y Value Equivalence"));
        std::vector<double> single{42.0};
        const double single_med = median(single.begin(), single.end());
        output_test_results.push_back(run_test(42.0, single_med, "Median should be 42.0 for a single-element vector.", "Median, Single Element"));

        check_tests(output_test_results);
    } 
    catch(const std::exception &e) 
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "All tests passed successfully." << std::endl;
    return 0;
}