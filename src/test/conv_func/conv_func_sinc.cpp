/** @file conv_func_sinc.cpp
 *  @brief Test Sinc
 *
 *  TestCase to test the sinc convolution function
 *  test with array input.
 *
 *  @bug No known bugs.
 */

#include "../../libstp/convolution/conv_func.h"
#include "gtest/gtest.h"

// Test the sinc functor implementation.
TEST(ConvSincFunc, conv_funcs_sinc)  {
    Sinc sinc;

    arma::mat input = { 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5 };
    arma::mat output =
        {
            1.0,
            1. / (0.5 * arma::datum::pi),
            0.0,
            -1. / (1.5 * arma::datum::pi),
            0.0,
            1. / (2.5 * arma::datum::pi),
            0.0
        };

    EXPECT_TRUE(arma::approx_equal(sinc(input), output, "absdiff", 0.091));
}
