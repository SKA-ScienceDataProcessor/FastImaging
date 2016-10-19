/** @file conv_func_gaussian.cpp
 *  @brief Test Gaussian
 *
 *  TestCase to test the gaussian convolution function.
 *
 *  @bug No known bugs.
 */

#include "../../libstp/convolution/conv_func.h"
#include "gtest/gtest.h"

const double tolerance(0.002);

TEST(ConvGaussianFunc, test_conv_funcs_test_gaussian) {
    mat input = { 0.0, 1.0, 3.1 };
    mat output = { 1.0, 1. /exp(1.), 0. };

    // test with array input
    EXPECT_TRUE(approx_equal(make_conv_func_gaussian(input, 1.0), output, "absdiff", tolerance));
}
