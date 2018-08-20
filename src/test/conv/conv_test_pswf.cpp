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
TEST(ConvPswfFunc, conv_funcs_pswf)
{
    arma::vec input = { 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 };

    /* From Python implementation:
     *     from fastimgproto.gridder.conv_funcs import PSWF
     *     kernel_support = 3
     *     kernel_func = PSWF(kernel_support)
     *     M = [0, 0.5, 1, 1.5, 2, 2.5, 3]
     *     kernel_func(M)
     *
     *     Out[8]:
     *     array([0.9999996673648565, 0.8729012518028004, 0.5732453907230994,
     *            0.27079903839833847, 0.08262342035850252, 0.012010540820816389, 0. ])
     */
    arma::vec output = {
        0.9999996673648565,
        0.8729012518028004,
        0.5732453907230994,
        0.27079903839833847,
        0.08262342035850252,
        0.012010540820816389,
        0.0
    };

    EXPECT_TRUE(arma::approx_equal(PSWF(3.0)(input), output, "absdiff", fptolerance));
}
