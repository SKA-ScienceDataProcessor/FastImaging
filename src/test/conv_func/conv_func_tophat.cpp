/** @file conv_func_tophat.cpp
 *  @brief Test TopHat
 *
 *  TestCase to test the tophat convolution function
 *  test with array input.
 *
 *  @bug No known bugs.
 */

#include "../../libstp/convolution/conv_func.h"
#include "gtest/gtest.h"

// Test the tophat functor implementation.
TEST(ConvTopHatFunc, conv_funcs_tophat_func) {
    TopHat tophat;
    const double half_base_width(3.0);

    arma::mat input = { 0.0, 2.5, 2.999, 3.0, 4.2 };
    arma::mat output = { 1.0, 1.0, 1.0, 0.0, 0.0 };

    EXPECT_TRUE(arma::approx_equal(tophat(input, half_base_width), output, "absdiff", tolerance));
}
