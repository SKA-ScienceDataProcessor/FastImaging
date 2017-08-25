/** @file conv_test_gaussian.cpp
 *  @brief Test Gaussian
 *
 *  TestCase to test the gaussian convolution function
 *  test with array input
 */

#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

// Test the gaussian functor implementation.
TEST(ConvGaussianFunc, test_conv_funcs_test_gaussian)
{
    arma::vec input = { 0.0, 1.0, 3.1 };
    arma::vec output = { 1.0, 1. / exp(1.), 0. };

    EXPECT_TRUE(arma::approx_equal(Gaussian(1.0, 3.0)(input), output, "absdiff", fptolerance));
}
