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

// Test the Matrix mean and stddev functions
TEST_F(MatrixMathTest, MatrixMeanStdDevFunction)
{
    double arma_mean = arma::mean(arma::vectorise(data));
    double arma_sigma = arma::stddev(arma::vectorise(data));
    auto d_stats = stp::mat_mean_and_stddev(data);

    EXPECT_NEAR(arma_mean, d_stats.mean, math_tolerance);
    EXPECT_NEAR(arma_sigma, d_stats.sigma, math_tolerance);
}
