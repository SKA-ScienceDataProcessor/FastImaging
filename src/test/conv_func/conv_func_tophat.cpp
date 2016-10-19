/** @file conv_func_tophat.cpp
 *  @brief Test TopHat
 *
 *  TestCase to test the tophat convolution function.
 *
 *  @bug No known bugs.
 */

#include "../../libstp/convolution/conv_func.h"
#include "gtest/gtest.h"

const double tolerance(0.002);

TEST(ConvTopHatFunc, conv_funcs_tophat_func) {
    double half_base_width(3.0);

    mat input = { 0.0, 2.5, 2.999, 3.0, 4.2 };
    mat output = { 1.0, 1.0, 1.0, 0.0, 0.0 };

    // test with array input
    EXPECT_TRUE(approx_equal(make_conv_func_tophat(half_base_width, input), output, "absdiff", tolerance));
}
