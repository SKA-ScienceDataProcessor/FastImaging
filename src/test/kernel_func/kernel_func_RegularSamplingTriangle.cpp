/** @file kernel_func_RegularSamplingTriangle.cpp
 *  @brief Test RegularSamplingTriangle
 *
 *  TestCase to test the kernel function in a
 *  regular sampling triangle example.
 *
 *  @bug No known bugs.
 */

#include <gtest/gtest.h>
#include <libstp.h>

const int support(2);
const double half_base_width(1.5);
const double oversampling(1);
const double triangle_value(1.0);
const bool pad(false);
const bool normalize(false);

const double kernel_value_at_1pix(1. - 1. / 1.5);
const double kv1(kernel_value_at_1pix);
const double kv1sq(kv1* kv1);
const double kv_half(1. - 0.5 / 1.5);

// Test 2D kernel convolution method in a regular sampling triangle, without offset.
TEST(KernelGenerationRegularSamplingTriangle, Offset1)
{
    arma::mat offset_index = { 0., 0. };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0. },
        { 0., kv1sq, kv1, kv1sq, 0. },
        { 0., kv1, 1., kv1, 0. },
        { 0., kv1sq, kv1, kv1sq, 0. },
        { 0., 0., 0., 0., 0. }
    };

    Triangle triangle(half_base_width, triangle_value);
    arma::mat result_array = make_kernel_array(support, offset_index, oversampling, pad, normalize, triangle);

    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", tolerance));
}

// Test 2D kernel convolution method in a regular sampling triangle, with .5 pix offset right.
TEST(KernelGenerationRegularSamplingTriangle, Offset2)
{
    arma::mat offset_index = { 0.5, 0. };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0. },
        { 0., 0., kv_half * kv1, kv_half * kv1, 0. },
        { 0., 0., kv_half, kv_half, 0. },
        { 0., 0., kv_half * kv1, kv_half * kv1, 0. },
        { 0., 0., 0., 0., 0. }
    };

    Triangle triangle(half_base_width, triangle_value);
    arma::mat result_array = make_kernel_array(support, offset_index, oversampling, pad, normalize, triangle);

    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", tolerance));
}
