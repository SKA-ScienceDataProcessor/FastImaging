/** @file kernel_func_OversampledPillboxsmall.cpp
 *  @brief Test OversampledPillboxSmall
 *
 *  TestCase to test the kernel function in an
 *  oversampled tophat-small example.
 *
 *  @bug No known bugs.
 */

#include "../../libstp/convolution/kernel_func.h"
#include "gtest/gtest.h"

const int support(1);
const double half_base_width(0.25);
const double oversampling(5);

const double tolerance(0.002);

TEST(KernelGenerationOversamplePillboxSmall, Offset1) {
    mat offset_index = { 0., 0. };

    mat expected_results =
        {{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.},
         {0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.},
         {0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}};

    mat result_array = make_top_hat_kernel_array(support, offset_index, oversampling, half_base_width, false,false);
    EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}

TEST(KernelGenerationOversamplePillboxSmall, Offset2) {
    mat offset_index = { 0.4, 0.0 };

    mat expected_results =
        {{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.},
         {0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.},
         {0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
         {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.}};

    mat result_array = make_top_hat_kernel_array(support, offset_index, oversampling, half_base_width, false,false);
    EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}
