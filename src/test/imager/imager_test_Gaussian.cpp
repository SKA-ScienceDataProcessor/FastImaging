/** @file imager_test_Gaussian.cpp
 *  @brief Test imager with Gaussian
 *
 *  TestCase to test the imager-imager
 *  implementation with gaussian convolution
 */

#include "load_json_imager.h"
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

struct imager_test_gaussian : public ImagerHandler {
    double width_normalization;
    double threshold;

public:
    imager_test_gaussian() = default;
    imager_test_gaussian(const std::string& typeConvolution, const std::string& typeTest)
        : ImagerHandler(typeConvolution, typeTest)
    {
        width_normalization = val["width_normalization"].GetDouble();
        threshold = val["threshold"].GetDouble();
        std::string expected_results_path = val["expected_results"].GetString();

        arma::cx_mat vis(load_npy_complex_array<double>(val["input_file"].GetString(), "vis"));
        arma::mat uvw_lambda(load_npy_double_array<double>(val["input_file"].GetString(), "uvw"));
        arma::mat vis_weights = arma::ones<arma::mat>(arma::size(vis));

        // Loads the expected results to a arma::mat pair
        expected_result = std::make_pair(std::move(load_npy_double_array<double>(expected_results_path, "image")), std::move(load_npy_double_array<double>(expected_results_path, "beam")));

        ImagerPars imgpars(image_size, cell_size, stp::KernelFunction::Gaussian, support, kernel_exact, oversampling, gen_beam, gridding_correction, analytic_gcf);
        std::pair<arma::Mat<real_t>, arma::Mat<real_t>> orig_result = image_visibilities(Gaussian(threshold, width_normalization), vis, vis_weights, uvw_lambda, imgpars);

#ifndef FFTSHIFT
        // Output matrices need to be shifted because image_visibilities does not shift them
        fftshift(orig_result.first);
        fftshift(orig_result.second);
#endif

        result.first = arma::conv_to<arma::mat>::from(orig_result.first);
        result.second = arma::conv_to<arma::mat>::from(orig_result.second);
    }
};

TEST(ImagerGaussian, SmallImage)
{
    imager_test_gaussian gaussian_small_image("gaussian", "small_image");
    EXPECT_TRUE(arma::approx_equal(gaussian_small_image.result.first, arma::real(gaussian_small_image.expected_result.first), "absdiff", fptolerance));
    EXPECT_TRUE(arma::approx_equal(gaussian_small_image.result.second, arma::real(gaussian_small_image.expected_result.second), "absdiff", fptolerance));
}

TEST(ImagerGaussian, MediumImage)
{
    imager_test_gaussian gaussian_medium_image("gaussian", "medium_image");
    EXPECT_TRUE(arma::approx_equal(gaussian_medium_image.result.first, arma::real(gaussian_medium_image.expected_result.first), "absdiff", fptolerance));
    EXPECT_TRUE(arma::approx_equal(gaussian_medium_image.result.second, arma::real(gaussian_medium_image.expected_result.second), "absdiff", fptolerance));
}

TEST(ImagerGaussian, LargeImage)
{
    imager_test_gaussian gaussian_large_image("gaussian", "large_image");
    EXPECT_TRUE(arma::approx_equal(gaussian_large_image.result.first, arma::real(gaussian_large_image.expected_result.first), "absdiff", fptolerance));
    EXPECT_TRUE(arma::approx_equal(gaussian_large_image.result.second, arma::real(gaussian_large_image.expected_result.second), "absdiff", fptolerance));
}
