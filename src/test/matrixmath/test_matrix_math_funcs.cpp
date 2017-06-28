/** @file conv_func_gaussian.cpp
 *  @brief Test Gaussian
 *
 *  TestCase to test the gaussian convolution function
 *  test with array input
 *
 *  @bug No known bugs.
 */

#include <cblas.h>
#include <common/matrix_math.h>
#include <fixtures.h>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

long size = 2048;

#ifdef USE_FLOAT
const double dtolerance = 1.0e-3;
#else
const double dtolerance = 1.0e-5;
#endif

// Test the Matrix accumulate function
TEST(MatrixAccumulateFunction, AccuValue)
{
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size);
    double arma_accu = arma::accu(arma::conv_to<arma::Mat<double>>::from(data));
    double m_accu = stp::mat_accumulate(data);

    EXPECT_NEAR(arma_accu, m_accu, dtolerance);
}

// Test the Matrix accumulate parallel function
TEST(MatrixAccumulateParallelFunction, ParAccuValue)
{
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size);
    double arma_accu = arma::accu(arma::conv_to<arma::Mat<double>>::from(data));
    double m_accu = stp::mat_accumulate_parallel(data);

    EXPECT_NEAR(arma_accu, m_accu, dtolerance);
}

// Test the Matrix mean function
TEST(MatrixMeanFunction, MeanValue)
{
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size);
    double arma_mean = arma::mean(arma::vectorise(data));
    double m_mean = stp::mat_mean(data);

    EXPECT_NEAR(arma_mean, m_mean, dtolerance);
}

// Test the Matrix mean parallel function
TEST(MatrixMeanParallelFunction, ParMeanValue)
{
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size);
    double arma_mean = arma::mean(arma::vectorise(data));
    double m_mean = stp::mat_mean_parallel(data);

    EXPECT_NEAR(arma_mean, m_mean, dtolerance);
}

// Test the Matrix stddev parallel function
TEST(MatrixStddevParallelFunction, ParStddevValue)
{
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size);
    double arma_std = arma::stddev(arma::vectorise(arma::conv_to<arma::Mat<double>>::from(data)));
    double m_std = stp::mat_stddev_parallel(data);

    EXPECT_NEAR(arma_std, m_std, dtolerance);
}

// Test matrix inplace division function
TEST(MatrixInplaceDivFunction, MatrixInplaceDiv)
{
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size);
    double a = 0.03549;

    arma::Mat<real_t> div_arma = data;
    arma::Mat<real_t> div_tbb = data;
    arma::Mat<real_t> div_cblas = data;

    // Arma inplace division
    div_arma /= a;

    // TBB inplace division
    size_t n_elem = div_tbb.n_elem;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_elem), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            div_tbb[i] /= a;
        }
    });

    // Cblas inplace division
    n_elem = div_cblas.n_elem;
#ifdef USE_FLOAT
    cblas_sscal((long)n_elem, (1.0f / a), div_cblas.memptr(), 1);
#else
    cblas_dscal(n_elem, (1.0 / a), div_cblas.memptr(), 1);
#endif

    EXPECT_TRUE(arma::approx_equal(div_arma, div_tbb, "absdiff", dtolerance));
    EXPECT_TRUE(arma::approx_equal(div_arma, div_cblas, "absdiff", dtolerance)); //We need to reduce tolerance value
}

// Test matrix shift function
TEST(MatrixShiftFunction, MatrixShift)
{
    arma::Mat<real_t> m = uncorrelated_gaussian_noise_background(size, size);
    arma::Mat<real_t> arma_out, matrix_out;

    arma_out = stp::matrix_shift(m, size / 2, 0);
    matrix_out = arma::shift(m, size / 2, 0);
    EXPECT_TRUE(arma::approx_equal(arma_out, matrix_out, "absdiff", 0));

    arma_out = stp::matrix_shift(m, size / 2, 1);
    matrix_out = arma::shift(m, size / 2, 1);
    EXPECT_TRUE(arma::approx_equal(arma_out, matrix_out, "absdiff", 0));
}

// Test the Matrix median function
TEST(MatrixMedianFunction, Median)
{
    double sigma = 0.5;
    double tolerance = sigma / 800; // In theory, the maximum error of binapprox median is sigma/1000
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, sigma, 0);
    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_median_binapprox(data);

    EXPECT_NEAR(arma_median, d_stats.median, tolerance);
}

// Test the Matrix mean and stddev functions
TEST(MatrixMeanStdDevFunction, MeanStdDev)
{
    double sigma = 0.5;
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, sigma);
    double arma_mean = arma::mean(arma::vectorise(data));
    double arma_sigma = arma::stddev(arma::vectorise(data));
    auto d_stats = stp::mat_mean_and_stddev(data);

    EXPECT_NEAR(arma_mean, d_stats.mean, dtolerance);
    EXPECT_NEAR(arma_sigma, d_stats.sigma, dtolerance);
}
