/** @file visibility_test_noise_generation.cpp
 *  @brief Test Visibility module implementation
 *
 *  @bug No known bugs.
 */

#include <cfloat>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class VisibilityNoiseGeneration : public ::testing::Test {
private:
    arma::cx_mat complex_zeroes;
    double noise_level;
    double seed;

public:
    void run()
    {
        complex_zeroes.set_size(1000000);
        complex_zeroes.zeros();
        noise_level = 1;
        seed = 1234;
        arma::cx_mat complex_noise_1jy = add_gaussian_noise(noise_level, complex_zeroes, seed);
        lower_mean_real = arma::mean(arma::vectorise(real(complex_noise_1jy)));
        lower_mean_img = arma::mean(arma::vectorise(imag(complex_noise_1jy)));
        std_dev_lower_real = abs(arma::stddev(arma::vectorise(real(complex_noise_1jy))) - 1.0);
        std_dev_lower_img = abs(arma::stddev(arma::vectorise(imag(complex_noise_1jy))) - 1.0);

        noise_level = 5;
        seed = 5678;
        arma::cx_mat complex_noise_5jy = add_gaussian_noise(noise_level, complex_zeroes, seed);
        large_mean_real = arma::mean(arma::vectorise(real(complex_noise_5jy)));
        large_mean_img = arma::mean(arma::vectorise(imag(complex_noise_5jy)));
        std_dev_large_real = abs(stddev(arma::vectorise(real(complex_noise_5jy))) - 5.0);
        std_dev_large_img = abs(stddev(arma::vectorise(imag(complex_noise_5jy))) - 5.0);

        EXPECT_LT(lower_mean_real, 0.001);
        EXPECT_LT(lower_mean_img, 0.001);
        EXPECT_LT(std_dev_lower_real, 0.02);
        EXPECT_LT(std_dev_lower_img, 0.02);
        EXPECT_LT(large_mean_real, 0.006);
        EXPECT_LT(large_mean_img, 0.006);
        EXPECT_LT(std_dev_large_real, 0.1);
        EXPECT_LT(std_dev_large_img, 0.1);
    }

    double lower_mean_real;
    double lower_mean_img;
    double std_dev_lower_real;
    double std_dev_lower_img;

    double large_mean_real;
    double large_mean_img;
    double std_dev_large_real;
    double std_dev_large_img;
};

// Check that the noise added has the expected properties for both real and complex components,
// by asserting mean / std.dev. are close to expected values.
TEST_F(VisibilityNoiseGeneration, test_visibility)
{
    run();
}
