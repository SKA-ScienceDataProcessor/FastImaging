/** @file kernel_test_RegularSamplingPillbox.cpp
 *  @brief Test RegularSamplingPillbox
 *
 *  TestCase to test the kernel function in a
 *  regular sampling tophat example.
 */

#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

const int support = 2;
const double half_base_width = 1.1;
const int oversampling = 1;
const bool pad = false;
const bool normalize = false;

using namespace stp;

// Test 2D kernel convolution method in a regular sampling pillbox, without offset.
TEST(KernelGenerationRegularSamplingPillbox, NoOffset)
{
    arma::mat offset_index = { 0., 0. };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0. },
        { 0., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 0. },
        { 0., 0., 0., 0., 0. }
    };

    arma::mat result_array = make_kernel_array(TopHat(half_base_width), support, offset_index, oversampling, pad, normalize);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", fptolerance));
}

// Test 2D kernel convolution method in a regular sampling pillbox, with tiny offest (less than pillbox overlap).
TEST(KernelGenerationRegularSamplingPillbox, SmallOffset)
{
    arma::mat offset_index = { 0.05, 0.05 };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0. },
        { 0., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 0. },
        { 0., 0., 0., 0., 0. }
    };

    arma::mat result_array = make_kernel_array(TopHat(half_base_width), support, offset_index, oversampling, pad, normalize);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", fptolerance));
}

// Test 2D kernel convolution method in a regular sampling pillbox; in this test shift the pillbox just enough to +ve x that we drop a column.
TEST(KernelGenerationRegularSamplingPillbox, Offset1)
{
    arma::mat offset_index = { 0.15, 0.0 };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0. },
        { 0., 0., 1., 1., 0. },
        { 0., 0., 1., 1., 0. },
        { 0., 0., 1., 1., 0. },
        { 0., 0., 0., 0., 0. }
    };

    arma::mat result_array = make_kernel_array<TopHat>(TopHat(half_base_width), support, offset_index, oversampling, pad, normalize);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", fptolerance));
}

// Test 2D kernel convolution method in a regular sampling pillbox; in this test shift the pillbox just enough to -ve x that we drop a column.
TEST(KernelGenerationRegularSamplingPillbox, Offset2)
{
    arma::mat offset_index = { -0.15, 0.0 };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0. },
        { 0., 1., 1., 0., 0. },
        { 0., 1., 1., 0., 0. },
        { 0., 1., 1., 0., 0. },
        { 0., 0., 0., 0., 0. }
    };

    arma::mat result_array = make_kernel_array<TopHat>(TopHat(half_base_width), support, offset_index, oversampling, pad, normalize);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", fptolerance));
}

// Test 2D kernel convolution method in a regular sampling pillbox; in this test shift the pillbox just enough to +ve y that we drop a column.
TEST(KernelGenerationRegularSamplingPillbox, Offset3)
{
    arma::mat offset_index = { 0.0, 0.15 };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0. },
        { 0., 1., 1., 1., 0. },
        { 0., 1., 1., 1., 0. },
        { 0., 0., 0., 0., 0. }
    };

    arma::mat result_array = make_kernel_array(TopHat(half_base_width), support, offset_index, oversampling, pad, normalize);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", fptolerance));
}

// Test 2D kernel convolution method in a regular sampling pillbox; in this test shift the pillbox just enough to +ve y & +ve x that we drop a column.
TEST(KernelGenerationRegularSamplingPillbox, Offset4)
{
    arma::mat offset_index = arma::mat{ 0.15, 0.15 };

    arma::mat expected_results = {
        { 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0. },
        { 0., 0., 1., 1., 0. },
        { 0., 0., 1., 1., 0. },
        { 0., 0., 0., 0., 0. }
    };

    arma::mat result_array = make_kernel_array(TopHat(half_base_width), support, offset_index, oversampling, pad, normalize);
    EXPECT_TRUE(arma::approx_equal(result_array, expected_results, "absdiff", fptolerance));
}
