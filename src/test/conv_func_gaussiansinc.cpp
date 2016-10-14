/** @file conv_func_gaussiansinc.cpp
 *  @brief Test GaussianSinc
 *
 *  TestCase to test the gaussiansinc convolution function.
 *
 *  @bug No known bugs.
 */

#include "../libstp/convolution/conv_func.h"
#include "gtest/gtest.h"

const double tolerance(0.002);

TEST(ConvGaussianSincFunc, test_conv_funcs_test_gaussian_sinc)  {
    double W1(2.52);
    double W2(1.55);

    mat input = { 0.0, W2 * 0.5, W2, W2 * 1.5, W2 * 2., W2 * 2.5, 5.5 };
     
    mat output = { 1.0,
                   exp(-1. * pow((0.5 * W2 / W1), 2.0)) * 1. / (0.5 * datum::pi),
                   0.0,
                   exp(-1. * pow((1.5 * W2 / W1), 2.0)) * -1. / (1.5 * datum::pi),
                   0.0,
                   exp(-1. * pow((2.5 * W2 / W1), 2.0)) * 1. / (2.5 * datum::pi),
                   0.0
                 };
    
    // test with array input
    EXPECT_TRUE(approx_equal(make_conv_func_gaussian_sinc(input, W1, W2), output, "absdiff", tolerance));
}
