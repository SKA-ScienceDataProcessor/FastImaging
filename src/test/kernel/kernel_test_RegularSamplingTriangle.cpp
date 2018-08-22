/** @file kernel_test_RegularSamplingTriangle.cpp
 *  @brief Test RegularSamplingTriangle
 *
 *  TestCase to test the kernel function in a
 *  regular sampling triangle example.
 */

#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

const int support = 2;
const double half_base_width = 1.5;
const int oversampling = 1;
const bool pad = false;
const bool normalize = false;

const real_t kv1(1. - 1. / 1.5);
const real_t kv1sq(kv1* kv1);
const real_t kv_half(1. - 0.5 / 1.5);

// Test 2D kernel convolution method in a regular sampling triangle, without offset.
TEST(KernelGenerationRegularSamplingTriangle, NoOffset)
{
    arma::mat offset_index = { 0., 0. };

    arma::Mat<real_t> expected_results = {
        { 0., 0., 0., 0., 0. },
        { 0., kv1sq, kv1, kv1sq, 0. },
        { 0., kv1, 1., kv1, 0. },
        { 0., kv1sq, kv1, kv1sq, 0. },
        { 0., 0., 0., 0., 0. }
    };

    arma::Mat<real_t> result_array = make_kernel_array(Triangle(half_base_width), support, offset_index, oversampling, pad, normalize);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", fptolerance));
}

// Test 2D kernel convolution method in a regular sampling triangle, with .5 pix offset right.
TEST(KernelGenerationRegularSamplingTriangle, OffsetRight)
{
    arma::mat offset_index = { 0.5, 0. };

    arma::Mat<real_t> expected_results = {
        { 0., 0., 0., 0., 0. },
        { 0., 0., kv_half * kv1, kv_half * kv1, 0. },
        { 0., 0., kv_half, kv_half, 0. },
        { 0., 0., kv_half * kv1, kv_half * kv1, 0. },
        { 0., 0., 0., 0., 0. }
    };

    arma::Mat<real_t> result_array = make_kernel_array(Triangle(half_base_width), support, offset_index, oversampling, pad, normalize);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", fptolerance));
}
