/** @file kernel_func_RegularSamplingTriangle.cpp
 *  @brief Test RegularSamplingTriangle
 *
 *  TestCase to test the kernel function in a
 *  regular sampling triangle example.
 *
 *  @bug No known bugs.
 */

#include "../libstp/convolution/kernel_func.h"
#include "gtest/gtest.h"

const int support(2);
const double half_base_width(1.5);
const double oversampling(1);

const double tolerance(0.002);

const double kernel_value_at_1pix(1. - 1. / 1.5);
const double kv1(kernel_value_at_1pix);
const double kv1sq(kv1 * kv1);
const double kv_half(1. - 0.5 / 1.5);

TEST(KernelGenerationRegularSamplingTriangle, Offset1) {
    mat offset_index = { 0., 0. };

    // No offset
    mat expected_results
        {{0., 0., 0., 0., 0.},
         {0., kv1sq, kv1, kv1sq, 0.},
         {0., kv1, 1., kv1, 0.},
         {0., kv1sq, kv1, kv1sq, 0.},
         {0., 0., 0., 0., 0.}};

    mat result_array = make_triangle_kernel_array(support, offset_index, oversampling, half_base_width);
    EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}

TEST(KernelGenerationRegularSamplingTriangle, Offset2) {
    mat offset_index = { 0.5, 0. };
    // .5 pix offset right:

    mat expected_results =
        {{0., 0., 0., 0., 0.},
         {0., 0., kv_half * kv1, kv_half * kv1, 0.},
         {0., 0., kv_half, kv_half, 0.},
         {0., 0., kv_half * kv1, kv_half * kv1, 0.},
         {0., 0., 0., 0., 0.}};

    mat result_array = make_triangle_kernel_array(support, offset_index, oversampling, half_base_width);
    EXPECT_TRUE(approx_equal(result_array,expected_results, "absdiff", tolerance));
}
