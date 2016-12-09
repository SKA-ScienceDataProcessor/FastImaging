/** @file kernel_func_OversampledPillboxsmall.cpp
 *  @brief Test OversampledPillboxSmall
 *
 *  TestCase to test the kernel function in an
 *  oversampled tophat-small example.
 *
 *  @bug No known bugs.
 */

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <libstp.h>

const int support(1);
const double half_base_width(0.25);
const double oversampling(5);
const bool pad(false);
const bool normalize(false);

// Test 2D kernel convolution method in an oversampled pillbox small, without offset.
TEST(KernelGenerationOversamplePillboxSmall, NoOffset)
{
    arma::mat offset_index = { 0., 0. };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }
    };

    arma::mat result_array = make_kernel_array(support, offset_index, oversampling, pad, normalize, TopHat(half_base_width));
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", tolerance));
}

// Test 2D kernel convolution method in an oversampled pillbox small, with .4 pix offset right.
TEST(KernelGenerationOversamplePillboxSmall, OffsetRight)
{
    arma::mat offset_index = { 0.4, 0.0 };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0. }
    };

    arma::mat result_array = make_kernel_array(support, offset_index, oversampling, pad, normalize, TopHat(half_base_width));
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", tolerance));
}

// Benchmark functions

void BM_OversamplePillboxSmall_without_offset(benchmark::State& state)
{
    arma::mat offset_index = { 0., 0. };

    while (state.KeepRunning())
        make_kernel_array(support, offset_index, oversampling, pad, normalize, TopHat(half_base_width));
}

TEST(KernelGenerationOversamplePillboxSmall, OversamplePillboxSmall_benchmark)
{
    BENCHMARK(BM_OversamplePillboxSmall_without_offset);
    benchmark::RunSpecifiedBenchmarks();
}
