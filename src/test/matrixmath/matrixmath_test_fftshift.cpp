#include <common/matrix_math.h>
#include <fixtures.h>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

const double math_tolerance = 1.0e-5;

// Test matrix shift function
TEST(MatrixMathFFTShiftTest, MatrixShiftCorrectness)
{
    long size = 512;
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size);

    arma::Mat<real_t> arma_out, matrix_out;

    arma_out = stp::matrix_shift(data, size / 2, 0);
    matrix_out = arma::shift(data, size / 2, 0);
    EXPECT_TRUE(arma::approx_equal(arma_out, matrix_out, "absdiff", 0));

    arma_out = stp::matrix_shift(data, size / 2, 1);
    matrix_out = arma::shift(data, size / 2, 1);
    EXPECT_TRUE(arma::approx_equal(arma_out, matrix_out, "absdiff", 0));
}
