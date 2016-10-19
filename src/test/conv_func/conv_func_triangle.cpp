/** @file conv_func_triangle.cpp
 *  @brief Test Triangle
 *
 *  TestCase to test the triangle convolution function.
 *
 *  @bug No known bugs.
 */

#include "../../libstp/convolution/conv_func.h"
#include "gtest/gtest.h"

const double tolerance(0.002);

TEST(ConvTriangleFunc, conv_funcs_triangle_func) {
    double half_base_width(2.0);
    double triangle_value(1.0);

    mat input = { 0.0, 1.0, 2.0, 2.000001, 100, 0.1, 0.5 };
    mat output = { 1.0, 0.5, 0.0, 0.0, 0.0, 0.95, 0.75 };

    // test with array input
    EXPECT_TRUE(approx_equal(make_conv_func_triangle(half_base_width, input, triangle_value), output, "absdiff", tolerance));
}
