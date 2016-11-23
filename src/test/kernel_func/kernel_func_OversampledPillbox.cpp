/** @file kernel_func_OversampledPillbox.cpp
 *  @brief Test OversampledPillbox
 *
 *  TestCase to test the kernel function in an
 *  oversampled tophat example.
 *
 *  @bug No known bugs.
 */

#include <gtest/gtest.h>
#include <libstp.h>

const int support(1);
const double half_base_width(0.7);
const double oversampling(3);
const bool pad(false);
const bool normalize(false);

// Test 2D kernel convolution method in an oversampled pillbox, without offset.
TEST(KernelGenerationOversampledPillbox, Offset1)
{
    arma::mat offset_index = { 0., 0. };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0., 0., 0. },
        { 0., 1., 1., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 1., 1., 0. },
        { 0., 0., 0., 0., 0., 0., 0. }
    };

    arma::mat result_array = make_kernel_array<TopHat>(support, offset_index, oversampling, pad, normalize, half_base_width);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", tolerance));
}

// Test 2D kernel convolution method in an oversampled pillbox, with a tiny offset.
TEST(KernelGenerationOversampledPillbox, Offset2)
{
    arma::mat offset_index = { 0.01, 0.01 };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0., 0., 0. },
        { 0., 1., 1., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 1., 1., 0. },
        { 0., 0., 0., 0., 0., 0., 0. }
    };

    arma::mat result_array = make_kernel_array<TopHat>(support, offset_index, oversampling, pad, normalize, half_base_width);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", tolerance));
}

// Test 2D kernel convolution method in an oversampled pillbox; we displace towards -ve x a bit: with -0.05 offset right.
TEST(KernelGenerationOversampledPillbox, Offset3)
{
    arma::mat offset_index = { -.05, 0. };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0., 0., 0. },
        { 0., 1., 1., 1., 1., 0., 0. },
        { 0., 1., 1., 1., 1., 0., 0. },
        { 0., 1., 1., 1., 1., 0., 0. },
        { 0., 1., 1., 1., 1., 0., 0. },
        { 0., 1., 1., 1., 1., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0. }
    };

    arma::mat result_array = make_kernel_array<TopHat>(support, offset_index, oversampling, pad, normalize, half_base_width);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", tolerance));
}

// Test 2D kernel convolution method in an oversampled pillbox, with 0.4 offset right.
TEST(KernelGenerationOversampledPillbox, Offset4)
{
    arma::mat offset_index = { 0.4, 0. };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 1., 1., 1., 1. },
        { 0., 0., 0., 1., 1., 1., 1. },
        { 0., 0., 0., 1., 1., 1., 1. },
        { 0., 0., 0., 1., 1., 1., 1. },
        { 0., 0., 0., 1., 1., 1., 1. },
        { 0., 0., 0., 0., 0., 0., 0. }
    };

    arma::mat result_array = make_kernel_array<TopHat>(support, offset_index, oversampling, pad, normalize, half_base_width);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", tolerance));
}
