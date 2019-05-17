/** @file imager_test_TopHat.cpp
 *  @brief Test imager with Tophat
 *
 *  TestCase to test the imager-imager
 *  implementation with tophat convolution
 */

#include "load_json_imager.h"
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

struct imager_test_tophat : public ImagerHandler {
    double half_base_width;

public:
    imager_test_tophat() = default;
    imager_test_tophat(const std::string& typeConvolution, const std::string& typeTest)
        : ImagerHandler(typeConvolution, typeTest)
    {
        half_base_width = val["half_base_width"].GetDouble();

        std::string expected_results_path = val["expected_results"].GetString();

        arma::cx_mat vis(load_npy_complex_array<double>(val["input_file"].GetString(), "vis"));
        arma::mat uvw_lambda(load_npy_double_array<double>(val["input_file"].GetString(), "uvw"));
        arma::mat vis_weights = arma::ones<arma::mat>(arma::size(vis));

        // Loads the expected results to a arma::mat pair
        expected_result = std::make_pair(std::move(load_npy_double_array<double>(expected_results_path, "image")), std::move(load_npy_double_array<double>(expected_results_path, "beam")));

        ImagerPars imgpars(image_size, cell_size, padding_factor, stp::KernelFunction::TopHat, support, kernel_exact, oversampling, gen_beam, gridding_correction, analytic_gcf);
        std::pair<arma::Mat<real_t>, arma::Mat<real_t>> orig_result = image_visibilities(TopHat(half_base_width), vis, vis_weights, uvw_lambda, imgpars);

#ifndef FFTSHIFT
        // Output matrices need to be shifted because image_visibilities does not shift them
        fftshift(orig_result.first);
        fftshift(orig_result.second);
#endif

        result.first = arma::conv_to<arma::mat>::from(orig_result.first);
        result.second = arma::conv_to<arma::mat>::from(orig_result.second);
    }
};

TEST(ImagerTopHat, SmallImage)
{
    imager_test_tophat tophat_small_image("tophat", "small_image");
    EXPECT_TRUE(arma::approx_equal(tophat_small_image.result.first, arma::real(tophat_small_image.expected_result.first), "absdiff", fptolerance));
    EXPECT_TRUE(arma::approx_equal(tophat_small_image.result.second, arma::real(tophat_small_image.expected_result.second), "absdiff", fptolerance));
}

TEST(ImagerTopHat, MediumImage)
{
    imager_test_tophat tophat_medium_image("tophat", "medium_image");
    EXPECT_TRUE(arma::approx_equal(tophat_medium_image.result.first, arma::real(tophat_medium_image.expected_result.first), "absdiff", fptolerance));
    EXPECT_TRUE(arma::approx_equal(tophat_medium_image.result.second, arma::real(tophat_medium_image.expected_result.second), "absdiff", fptolerance));
}

TEST(ImagerTopHat, LargeImage)
{
    imager_test_tophat tophat_large_image("tophat", "large_image");
    EXPECT_TRUE(arma::approx_equal(tophat_large_image.result.first, arma::real(tophat_large_image.expected_result.first), "absdiff", fptolerance));
    EXPECT_TRUE(arma::approx_equal(tophat_large_image.result.second, arma::real(tophat_large_image.expected_result.second), "absdiff", fptolerance));
}
