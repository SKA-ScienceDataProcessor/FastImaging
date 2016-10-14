/** @file kernel_func_RegularSamplingPillbox.cpp
 *  @brief Test RegularSamplingPillbox
 *
 *  TestCase to test the kernel function in a
 *  regular sampling tophat example.
 *
 *  @bug No known bugs.
 */

#include "../libstp/convolution/kernel_func.h"
#include "gtest/gtest.h"

const int support(2);
const double half_base_width(1.1);
const double oversampling(1);

const double tolerance(0.002);

TEST(KernelGenerationRegularSamplingPillbox, Offset1) {
    // Map subpixel offset to expected results
    mat offset_index = { 0., 0. };

    mat expected_results =
        {{0., 0., 0., 0., 0.},
         {0., 1., 1., 1., 0.},
         {0., 1., 1., 1., 0.},
         {0., 1., 1., 1., 0.},
         {0., 0., 0., 0., 0.}};

    mat result_array = make_top_hat_kernel_array(support, offset_index, oversampling, half_base_width);
	EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}

TEST(KernelGenerationRegularSamplingPillbox, Offset2) {
    // Map subpixel offset to expected results
    mat offset_index = { 0.05, 0.05 };

    // Tiny offset (less than pillbox overlap) - expect same:
    mat expected_results =
        {{0., 0., 0., 0., 0.},
         {0., 1., 1., 1., 0.},
         {0., 1., 1., 1., 0.},
         {0., 1., 1., 1., 0.},
         {0., 0., 0., 0., 0.}};

    mat result_array = make_top_hat_kernel_array(support, offset_index, oversampling, half_base_width);
	EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}

TEST(KernelGenerationRegularSamplingPillbox, Offset3) {
    // Map subpixel offset to expected results
    mat offset_index = { 0.15, 0.0 };

    // Now shift the pillbox just enough to +ve x that we drop a column
    mat expected_results =
        {{0., 0., 0., 0., 0.},
         {0., 0., 1., 1., 0.},
         {0., 0., 1., 1., 0.},
         {0., 0., 1., 1., 0.},
         {0., 0., 0., 0., 0.}};

    mat result_array = make_top_hat_kernel_array(support, offset_index, oversampling, half_base_width);
	EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}

TEST(KernelGenerationRegularSamplingPillbox, Offset4) {
    // Map subpixel offset to expected results
    mat offset_index = { -0.15, 0.0 };

    // And shift to -ve x:
    mat expected_results =
        {{0., 0., 0., 0., 0.},
         {0., 1., 1., 0., 0.},
         {0., 1., 1., 0., 0.},
         {0., 1., 1., 0., 0.},
         {0., 0., 0., 0., 0.}};

    mat result_array = make_top_hat_kernel_array(support, offset_index, oversampling, half_base_width);
	EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}

TEST(KernelGenerationRegularSamplingPillbox, Offset5) {
    // Map subpixel offset to expected results
    mat offset_index = { 0.0, 0.15 };

    // Shift to +ve y:
    mat expected_results =
        {{0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0.},
         {0., 1., 1., 1., 0.},
         {0., 1., 1., 1., 0.},
         {0., 0., 0., 0., 0.}};

    mat result_array = make_top_hat_kernel_array(support, offset_index, oversampling, half_base_width);
	EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}

TEST(KernelGenerationRegularSamplingPillbox, Offset6) {
    // Map subpixel offset to expected results
    mat offset_index = mat { 0.15, 0.15 };

    // Shift to +ve y & +ve x:
    mat expected_results =
        {{0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0.},
         {0., 0., 1., 1., 0.},
         {0., 0., 1., 1., 0.},
         {0., 0., 0., 0., 0.}};

    mat result_array = make_top_hat_kernel_array(support, offset_index, oversampling, half_base_width);
	EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}
