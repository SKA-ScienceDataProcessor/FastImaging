/** @file conv_func_sinc.cpp
 *  @brief Test Sinc
 *
 *  TestCase to test the sinc convolution function.
 *
 *  @bug No known bugs.
 */

#include "../libstp/convolution/conv_func.h"
#include "gtest/gtest.h"

const double tolerance(0.091);

TEST(ConvSincFunc, conv_funcs_sinc)  {
    mat input = { 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5 };
    mat output = { 1.0, 1. / (0.5 * datum::pi), 0.0, -1. / (1.5 * datum::pi), 0.0, 1. / (2.5 * datum::pi), 0.0 };

    // test with array input
    EXPECT_TRUE(approx_equal(make_conv_func_sinc(input), output, "absdiff", tolerance));
}
