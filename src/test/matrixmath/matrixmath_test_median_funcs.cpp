#include <common/matrix_math.h>
#include <fixtures.h>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

long size = 512;

const double median_tolerance = 1.0e-12;

// Test the Matrix exact median function
TEST(MatrixMedianExactFunction, TestMedian)
{
    double sigma = 1.0;
    double mean = 0.0;
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, sigma, mean, 0);
    double arma_median = arma::median(arma::vectorise(data));
    double stp_median_exact = mat_median_exact(data);
    EXPECT_NEAR(arma_median, stp_median_exact, median_tolerance);
}

// Test the Matrix approximate median function
TEST(MatrixApproxMedianFunction, TestMedian)
{
    double sigma = 0.5;
    double mean = 0.0;
    double tolerance = sigma / 800.0; // In theory, the maximum error of binapprox median is sigma/1000
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, sigma, mean, 0);
    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_median_binapprox(data);

    EXPECT_NEAR(arma_median, d_stats.median, tolerance);
}

TEST(MatrixApproxMedianFunction, TestMedianSpecificPattern1)
{
    arma::Mat<real_t> data(20, 20);
    data.fill(100.0);
    data.cols(0, 9).fill(-100.0);
    data.at(0, 19) = 200.0;
    data.at(19, 0) = -200.0;
    double tolerance = 100.0 / 800.0;

    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_median_binapprox(data);
    EXPECT_NEAR(arma_median, d_stats.median, tolerance);
}

TEST(MatrixApproxMedianFunction, TestMedianSpecificPattern2)
{
    arma::Mat<real_t> data(20, 20);
    data.fill(100);
    data.cols(0, 9).fill(-100);
    double tolerance = 100.0 / 800.0;

    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_median_binapprox(data);
    EXPECT_NEAR(arma_median, d_stats.median, tolerance);
}

// Test the Matrix binmedian function
TEST(MatrixBinMedianFunction, TestMedian)
{
    double sigma = 1.0;
    double mean = 0.0;
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, sigma, mean, 0);
    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_binmedian(data);
    EXPECT_NEAR(arma_median, d_stats.median, median_tolerance);
}

TEST(MatrixBinMedianFunction, TestMedianWithVariableSeed)
{
    double sigma = 1.0;
    double mean = 0.0;
    for (int i = 0; i < 10; ++i) {
        arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, sigma, mean, i);
        double arma_median = arma::median(arma::vectorise(data));
        auto d_stats = mat_binmedian(data);
        EXPECT_NEAR(arma_median, d_stats.median, median_tolerance);
    }
}

TEST(MatrixBinMedianFunction, TestMedianWithVariableMean)
{
    double sigma = 1.0;
    for (int i = -50; i < 50; i += 25) {
        arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, sigma, i, 0);
        double arma_median = arma::median(arma::vectorise(data));
        auto d_stats = mat_binmedian(data);
        EXPECT_NEAR(arma_median, d_stats.median, median_tolerance);
    }
}

TEST(MatrixBinMedianFunction, TestMedianWithVariableSigma)
{
    double mean = 0.0;
    for (int i = 0; i < 50; i += 10) {
        arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, i, mean, 0);
        double arma_median = arma::median(arma::vectorise(data));
        auto d_stats = mat_binmedian(data);
        EXPECT_NEAR(arma_median, d_stats.median, median_tolerance);
    }
}

TEST(MatrixBinMedianFunction, TestMedianWithLargeSigma)
{
    double sigma = 1.0e10;
    double mean = 0.0;
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, sigma, mean, 0);
    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_binmedian(data);
    EXPECT_NEAR(arma_median, d_stats.median, median_tolerance);
}

TEST(MatrixBinMedianFunction, TestMedianWithSmallSigma)
{
    double sigma = 1.0e-4;
    double mean = 0.0;
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, sigma, mean, 0);
    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_binmedian(data);
    EXPECT_NEAR(arma_median, d_stats.median, median_tolerance);
}

TEST(MatrixBinMedianFunction, TestMedianSpecificPattern1)
{
    arma::Mat<real_t> data = {
        { 1, 30, 60 },
        { 1, 30, 60 },
        { 1, 30, 60 }
    };
    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_binmedian(data);
    EXPECT_NEAR(arma_median, d_stats.median, median_tolerance);
}

TEST(MatrixBinMedianFunction, TestMedianSpecificPattern2)
{
    arma::Mat<real_t> data = {
        { 1, 1, 60 },
        { 1, 1, 60 },
        { 1, 60, 60 }
    };
    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_binmedian(data);
    EXPECT_NEAR(arma_median, d_stats.median, median_tolerance);
}

TEST(MatrixBinMedianFunction, TestMedianSpecificPattern3)
{
    arma::Mat<real_t> data = {
        { 1, 1, 1 },
        { 1, 1, 1 },
        { 1, 1, 60 }
    };
    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_binmedian(data);
    EXPECT_NEAR(arma_median, d_stats.median, median_tolerance);
}

TEST(MatrixBinMedianFunction, TestMedianSpecificPattern4)
{
    arma::Mat<real_t> data(20, 20);
    data.fill(100.0);
    data.cols(0, 9).fill(-100.0);

    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_binmedian(data);
    EXPECT_NEAR(arma_median, d_stats.median, median_tolerance);
}

TEST(MatrixBinMedianFunction, TestMedianSpecificPattern5)
{
    arma::Mat<real_t> data(20, 20);
    data.fill(100.0);
    data.cols(0, 9).fill(-100);
    data.at(0, 19) = 200.0;
    data.at(19, 0) = -200.0;

    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_binmedian(data);
    EXPECT_NEAR(arma_median, d_stats.median, median_tolerance);
}

TEST(MatrixBinMedianFunction, TestMedianSpecificPattern6)
{
    double sigma = 1.0;
    double mean = 0.0;
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(20, 20, sigma, mean, 0);
    data[0] = 10000.0;
    data[1] = -10000.0;

    double arma_median = arma::median(arma::vectorise(data));
    auto d_stats = mat_binmedian(data);
    EXPECT_NEAR(arma_median, d_stats.median, median_tolerance);
}
