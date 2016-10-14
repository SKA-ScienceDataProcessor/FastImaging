/** @file kernel_func_OversampledPillbox.cpp
 *  @brief Test OversampledPillbox
 *
 *  TestCase to test the kernel function in an
 *  oversampled tophat example.
 *
 *  @bug No known bugs.
 */

#include "../libstp/convolution/kernel_func.h"
#include "gtest/gtest.h"

const int support(1);
const double half_base_width(0.7);
const double oversampling(3);

const double tolerance(0.002);

TEST(KernelGenerationOversampledPillbox, Offset1) {
    mat offset = { 0., 0. };

    mat expected_results =
        {{0., 0., 0., 0., 0., 0., 0.},
         {0., 1., 1., 1., 1., 1., 0.},
         {0., 1., 1., 1., 1., 1., 0.},
         {0., 1., 1., 1., 1., 1., 0.},
         {0., 1., 1., 1., 1., 1., 0.},
         {0., 1., 1., 1., 1., 1., 0.},
         {0., 0., 0., 0., 0., 0., 0.}};

    mat result_array = make_top_hat_kernel_array(support, offset, oversampling, half_base_width);
    EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}

TEST(KernelGenerationOversampledPillbox, Offset2) {
    mat offset = {0.01, 0.01};

    mat expected_results =
        {{0., 0., 0., 0., 0., 0., 0.},
         {0., 1., 1., 1., 1., 1., 0.},
         {0., 1., 1., 1., 1., 1., 0.},
         {0., 1., 1., 1., 1., 1., 0.},
         {0., 1., 1., 1., 1., 1., 0.},
         {0., 1., 1., 1., 1., 1., 0.},
         {0., 0., 0., 0., 0., 0., 0.}};

    mat result_array = make_top_hat_kernel_array(support, offset, oversampling, half_base_width);
    EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}

TEST(KernelGenerationOversampledPillbox, Offset3) {
    mat offset = { -.05, 0. };

    mat expected_results =
        {{0., 0., 0., 0., 0., 0., 0.},
         {0., 1., 1., 1., 1., 0., 0.},
         {0., 1., 1., 1., 1., 0., 0.},
         {0., 1., 1., 1., 1., 0., 0.},
         {0., 1., 1., 1., 1., 0., 0.},
         {0., 1., 1., 1., 1., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0.}};

    mat result_array = make_top_hat_kernel_array(support, offset, oversampling, half_base_width);
    EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}

TEST(KernelGenerationOversampledPillbox, Offset4) {
    mat offset = { 0.4, 0. };

    mat expected_results =
        {{0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 1., 1., 1., 1.},
         {0., 0., 0., 1., 1., 1., 1.},
         {0., 0., 0., 1., 1., 1., 1.},
         {0., 0., 0., 1., 1., 1., 1.},
         {0., 0., 0., 1., 1., 1., 1.},
         {0., 0., 0., 0., 0., 0., 0.}};

    mat result_array = make_top_hat_kernel_array(support, offset, oversampling, half_base_width);
    EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}
