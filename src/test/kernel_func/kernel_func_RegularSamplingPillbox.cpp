/** @file kernel_func_RegularSamplingPillbox.cpp
 *  @brief Test RegularSamplingPillbox
 *
 *  TestCase to test the kernel function in a
 *  regular sampling tophat example.
 *
 *  @bug No known bugs.
 */

#include <libstp.h>
#include <gtest/gtest.h>

const int support(2);
const double half_base_width(1.1);
const double oversampling(1);
const bool pad(false);
const bool normalize(false);

// Test 2D kernel convolution method in a regular sampling pillbox, without offset.
TEST(KernelGenerationRegularSamplingPillbox, Offset1) {
    arma::mat offset_index = { 0., 0. };

    arma::mat expected_results =
        {
            { 0., 0., 0., 0., 0. },
            { 0., 1., 1., 1., 0. },
            { 0., 1., 1., 1., 0. },
            { 0., 1., 1., 1., 0. },
            { 0., 0., 0., 0., 0. }
        };

    arma::mat result_array = make_kernel_array<TopHat>(support, offset_index, oversampling, pad, normalize, half_base_width);
	EXPECT_TRUE(arma::approx_equal(result_array,expected_results, "absdiff", tolerance));
}

// Test 2D kernel convolution method in a regular sampling pillbox, with tiny offest (less than pillbox overlap).
TEST(KernelGenerationRegularSamplingPillbox, Offset2) {
    arma::mat offset_index = { 0.05, 0.05 };

    arma::mat expected_results =
        {
            { 0., 0., 0., 0., 0. },
            { 0., 1., 1., 1., 0. },
            { 0., 1., 1., 1., 0. },
            { 0., 1., 1., 1., 0. },
            { 0., 0., 0., 0., 0. }
        };

    arma::mat result_array = make_kernel_array<TopHat>(support, offset_index, oversampling, pad, normalize, half_base_width);
	EXPECT_TRUE(arma::approx_equal(result_array,expected_results, "absdiff", tolerance));
}

// Test 2D kernel convolution method in a regular sampling pillbox; in this test shift the pillox just enought to +ve x that we drop a column.
TEST(KernelGenerationRegularSamplingPillbox, Offset3) {
    arma::mat offset_index = { 0.15, 0.0 };

    arma::mat expected_results =
        {
            { 0., 0., 0., 0., 0. },
            { 0., 0., 1., 1., 0. },
            { 0., 0., 1., 1., 0. },
            { 0., 0., 1., 1., 0. },
            { 0., 0., 0., 0., 0. }
        };

    arma::mat result_array = make_kernel_array<TopHat>(support, offset_index, oversampling, pad, normalize, half_base_width);
	EXPECT_TRUE(arma::approx_equal(result_array,expected_results, "absdiff", tolerance));
}

// Test 2D kernel convolution method in a regular sampling pillbox; in this test shift the pillox just enought to -ve x that we drop a column.
TEST(KernelGenerationRegularSamplingPillbox, Offset4) {
    arma::mat offset_index = { -0.15, 0.0 };

    arma::mat expected_results =
        {
            { 0., 0., 0., 0., 0. },
            { 0., 1., 1., 0., 0. },
            { 0., 1., 1., 0., 0. },
            { 0., 1., 1., 0., 0. },
            { 0., 0., 0., 0., 0. }
        };

    arma::mat result_array = make_kernel_array<TopHat>(support, offset_index, oversampling, pad, normalize, half_base_width);
	EXPECT_TRUE(arma::approx_equal(result_array,expected_results, "absdiff", tolerance));
}

// Test 2D kernel convolution method in a regular sampling pillbox; in this test shift the pillox just enought to +ve y that we drop a column.
TEST(KernelGenerationRegularSamplingPillbox, Offset5) {
    arma::mat offset_index = { 0.0, 0.15 };

    arma::mat expected_results =
        {
            { 0., 0., 0., 0., 0. },
            { 0., 0., 0., 0., 0. },
            { 0., 1., 1., 1., 0. },
            { 0., 1., 1., 1., 0. },
            { 0., 0., 0., 0., 0. }
        };

    arma::mat result_array = make_kernel_array<TopHat>(support, offset_index, oversampling, pad, normalize, half_base_width);
	EXPECT_TRUE(arma::approx_equal(result_array,expected_results, "absdiff", tolerance));
}

// Test 2D kernel convolution method in a regular sampling pillbox; in this test shift the pillox just enought to +ve y & +ve x that we drop a column.
TEST(KernelGenerationRegularSamplingPillbox, Offset6) {
    arma::mat offset_index = arma::mat { 0.15, 0.15 };

    arma::mat expected_results =
        {
            { 0., 0., 0., 0., 0. },
            { 0., 0., 0., 0., 0. },
            { 0., 0., 1., 1., 0. },
            { 0., 0., 1., 1., 0. },
            { 0., 0., 0., 0., 0. }
        };

    arma::mat result_array = make_kernel_array<TopHat>(support, offset_index, oversampling, pad, normalize, half_base_width);
	EXPECT_TRUE(arma::approx_equal(result_array,expected_results, "absdiff", tolerance));
}
