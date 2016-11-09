/** @file conv_func_gaussian.cpp
 *  @brief Test Gaussian
 *
 *  TestCase to test the gaussian convolution function
 *  test with array input
 *
 *  @bug No known bugs.
 */

#include <libstp.h>
#include <gtest/gtest.h>

// Test the gaussian functor implementation.
TEST(ConvGaussianFunc, test_conv_funcs_test_gaussian) {
    Gaussian gaussian;
    const double width(1.0);

    arma::mat input = { 0.0, 1.0, 3.1 };
    arma::mat output = { 1.0, 1. / exp(1.), 0. };

    EXPECT_TRUE(arma::approx_equal(gaussian(input, width), output, "absdiff", tolerance));
}
