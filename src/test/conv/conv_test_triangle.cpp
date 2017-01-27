/** @file conv_func_triangle.cpp
 *  @brief Test Triangle
 *
 *  TestCase to test the triangle convolution function
 *  test with array input.
 *
 *  @bug No known bugs.
 */

#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

// Test the triangle functor implementation.
TEST(ConvTriangleFunc, conv_funcs_triangle_func)
{
    arma::mat input = { 0.0, 1.0, 2.0, 2.000001, 100, 0.1, 0.5 };
    arma::mat output = { 1.0, 0.5, 0.0, 0.0, 0.0, 0.95, 0.75 };

    EXPECT_TRUE(arma::approx_equal(Triangle(2.0)(input), output, "absdiff", tolerance));
}
