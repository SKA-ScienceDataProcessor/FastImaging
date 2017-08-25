#include <cblas.h>
#include <common/matrix_math.h>
#include <fixtures.h>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

const double math_tolerance = 1.0e-5;

// Fixture class that builds matrix data for each unit test
class MatrixMathTest : public ::testing::Test {
public:
    void SetUp()
    {
        size = 512;
        data = uncorrelated_gaussian_noise_background(size, size);
    }
    arma::Mat<real_t> data;
    long size;
};

// Test the Matrix accumulate function
TEST_F(MatrixMathTest, MatrixAccumulateFunction)
{
    double arma_accu = arma::accu(arma::conv_to<arma::Mat<double>>::from(data));
    double m_accu = stp::mat_accumulate(data);

    EXPECT_NEAR(arma_accu, m_accu, math_tolerance);
}

// Test the Matrix accumulate parallel function
TEST_F(MatrixMathTest, MatrixAccumulateParallelFunction)
{
    double arma_accu = arma::accu(arma::conv_to<arma::Mat<double>>::from(data));
    double m_accu = stp::mat_accumulate_parallel(data);

    EXPECT_NEAR(arma_accu, m_accu, math_tolerance);
}

// Test the Matrix mean function
TEST_F(MatrixMathTest, MatrixMeanFunction)
{
    double arma_mean = arma::mean(arma::vectorise(data));
    double m_mean = stp::mat_mean(data);

    EXPECT_NEAR(arma_mean, m_mean, math_tolerance);
}

// Test the Matrix mean parallel function
TEST_F(MatrixMathTest, MatrixMeanParallelFunction)
{
    double arma_mean = arma::mean(arma::vectorise(data));
    double m_mean = stp::mat_mean_parallel(data);

    EXPECT_NEAR(arma_mean, m_mean, math_tolerance);
}

// Test the Matrix stddev parallel function
TEST_F(MatrixMathTest, MatrixStddevParallelFunction)
{
    double arma_std = arma::stddev(arma::vectorise(arma::conv_to<arma::Mat<double>>::from(data)));
    double m_std = stp::mat_stddev_parallel(data);

    EXPECT_NEAR(arma_std, m_std, math_tolerance);
}

// Test matrix inplace division function
TEST_F(MatrixMathTest, MatrixInplaceDivFunction)
{
    double a = 0.3549;

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

    EXPECT_TRUE(arma::approx_equal(div_arma, div_tbb, "absdiff", math_tolerance));
    EXPECT_TRUE(arma::approx_equal(div_arma, div_cblas, "absdiff", math_tolerance)); //We need to reduce tolerance value
}

// Test matrix shift function
TEST_F(MatrixMathTest, MatrixShiftFunction)
{
    arma::Mat<real_t> arma_out, matrix_out;

    arma_out = stp::matrix_shift(data, size / 2, 0);
    matrix_out = arma::shift(data, size / 2, 0);
    EXPECT_TRUE(arma::approx_equal(arma_out, matrix_out, "absdiff", 0));

    arma_out = stp::matrix_shift(data, size / 2, 1);
    matrix_out = arma::shift(data, size / 2, 1);
    EXPECT_TRUE(arma::approx_equal(arma_out, matrix_out, "absdiff", 0));
}

// Test the Matrix mean and stddev functions
TEST_F(MatrixMathTest, MatrixMeanStdDevFunction)
{
    double arma_mean = arma::mean(arma::vectorise(data));
    double arma_sigma = arma::stddev(arma::vectorise(data));
    auto d_stats = stp::mat_mean_and_stddev(data);

    EXPECT_NEAR(arma_mean, d_stats.mean, math_tolerance);
    EXPECT_NEAR(arma_sigma, d_stats.sigma, math_tolerance);
}
