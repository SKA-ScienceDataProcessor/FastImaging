/** @file conv_func_triangle.cpp
 *  @brief Test Triangle
 *
 *  TestCase to test the triangle convolution function
 *  test with array input.
 *
 *  @bug No known bugs.
 */

#include "../../libstp/convolution/conv_func.h"
#include "gtest/gtest.h"

// Test the triangle functor implementation.
TEST(ConvTriangleFunc, conv_funcs_triangle_func) {
    Triangle triangle;
    const double half_base_width(2.0);
    const double triangle_value(1.0);    

    arma::mat input = { 0.0, 1.0, 2.0, 2.000001, 100, 0.1, 0.5 };
    arma::mat output = { 1.0, 0.5, 0.0, 0.0, 0.0, 0.95, 0.75 };

    EXPECT_TRUE(arma::approx_equal(triangle(input, half_base_width, triangle_value), output, "absdiff", tolerance));
}
