/** @file conv_test_sinc.cpp
 *  @brief Test Sinc
 *
 *  TestCase to test the sinc convolution function
 *  test with array input.
 */

#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

// Test the sinc functor implementation.
TEST(ConvSincFunc, conv_funcs_sinc)
{
    arma::vec input = { 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5 };
    arma::vec output = {
        1.0,
        1. / (0.5 * arma::datum::pi),
        0.0,
        -1. / (1.5 * arma::datum::pi),
        0.0,
        1. / (2.5 * arma::datum::pi),
        0.0
    };

    EXPECT_TRUE(arma::approx_equal(Sinc(3.0, 1.0)(input), output, "absdiff", fptolerance));
}
