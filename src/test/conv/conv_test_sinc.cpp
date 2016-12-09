/** @file conv_func_sinc.cpp
 *  @brief Test Sinc
 *
 *  TestCase to test the sinc convolution function
 *  test with array input.
 *
 *  @bug No known bugs.
 */

#include <gtest/gtest.h>
#include <libstp.h>

// Test the sinc functor implementation.
TEST(ConvSincFunc, conv_funcs_sinc)
{
    // width_normalization = 1.0
    // threshold = 3.0

    arma::mat input = { 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5 };
    arma::mat output = {
        1.0,
        1. / (0.5 * arma::datum::pi),
        0.0,
        -1. / (1.5 * arma::datum::pi),
        0.0,
        1. / (2.5 * arma::datum::pi),
        0.0
    };

    EXPECT_TRUE(arma::approx_equal(Sinc(1.0, 3.0)(input), output, "absdiff", tolerance));
}
