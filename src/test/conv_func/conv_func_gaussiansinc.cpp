/** @file conv_func_gaussiansinc.cpp
 *  @brief Test GaussianSinc
 *
 *  TestCase to test the gaussiansinc convolution function
 *  test with array input.
 *
 *  @bug No known bugs.
 */

#include <gtest/gtest.h>
#include <libstp.h>

// Test the gaussiansinc functor implementation; with conventional scaling values.
TEST(ConvGaussianSincFunc, test_conv_funcs_test_gaussian_sinc)
{
    const double width_normalization_gaussian(2.52);
    const double width_normalization_sinc(1.55);
    const double trunc(5.0);
    GaussianSinc gaussian_sinc(width_normalization_gaussian, width_normalization_sinc, trunc);

    arma::mat input = {
        0.0,
        width_normalization_sinc * 0.5,
        width_normalization_sinc,
        width_normalization_sinc * 1.5,
        width_normalization_sinc * 2.,
        width_normalization_sinc * 2.5,
        5.5
    };

    arma::mat output = {
        1.0,
        exp(-1. * pow((0.5 * width_normalization_sinc / width_normalization_gaussian), 2.0)) * 1. / (0.5 * arma::datum::pi),
        0.0,
        exp(-1. * pow((1.5 * width_normalization_sinc / width_normalization_gaussian), 2.0)) * -1. / (1.5 * arma::datum::pi),
        0.0,
        exp(-1. * pow((2.5 * width_normalization_sinc / width_normalization_gaussian), 2.0)) * 1. / (2.5 * arma::datum::pi),
        0.0
    };

    EXPECT_TRUE(arma::approx_equal(gaussian_sinc(input), output, "absdiff", tolerance));
}
