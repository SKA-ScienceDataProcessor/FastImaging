/** @file kernel_test_OversampledPillboxsmall.cpp
 *  @brief Test OversampledPillboxSmall
 *
 *  TestCase to test the kernel function in an
 *  oversampled tophat-small example.
 */

#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

const int support = 1;
const double half_base_width = 0.25;
const int oversampling = 5;
const bool pad = false;
const bool normalize = false;

// Test 2D kernel convolution method in an oversampled pillbox small, without offset.
TEST(KernelGenerationOversamplePillboxSmall, NoOffset)
{
    arma::mat offset_index = { 0., 0. };

    arma::Mat<real_t> expected_results = {
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

    arma::Mat<real_t> result_array = make_kernel_array(TopHat(half_base_width), support, offset_index, oversampling, pad, normalize);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", fptolerance));
}

// Test 2D kernel convolution method in an oversampled pillbox small, with .4 pix offset right.
TEST(KernelGenerationOversamplePillboxSmall, OffsetRight)
{
    arma::mat offset_index = { 0.4, 0.0 };

    arma::Mat<real_t> expected_results = {
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

    arma::Mat<real_t> result_array = make_kernel_array(TopHat(half_base_width), support, offset_index, oversampling, pad, normalize);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", fptolerance));
}
