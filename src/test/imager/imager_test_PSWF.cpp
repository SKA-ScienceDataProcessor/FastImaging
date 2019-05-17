/** @file imager_test_GaussianSinc.cpp
 *  @brief Test imager with GaussianSinc
 *
 *  TestCase to test the imager-imager
 *  implementation with gaussiansinc convolution
 */

#include "load_json_imager.h"
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

struct imager_test_pswf : public ImagerHandler {
    double trunc;

public:
    imager_test_pswf() = default;
    imager_test_pswf(const std::string& typeConvolution, const std::string& typeTest)
        : ImagerHandler(typeConvolution, typeTest)
    {
        trunc = val["trunc"].GetDouble();
        std::string expected_results_path = val["expected_results"].GetString();

        arma::cx_mat vis(load_npy_complex_array<double>(val["input_file"].GetString(), "vis"));
        arma::mat uvw_lambda(load_npy_double_array<double>(val["input_file"].GetString(), "uvw"));
        arma::mat vis_weights = arma::ones<arma::mat>(arma::size(vis));

        // Loads the expected results to a arma::Mat pair
        expected_result = std::make_pair(std::move(load_npy_double_array<double>(expected_results_path, "image")), std::move(load_npy_double_array<double>(expected_results_path, "beam")));

        ImagerPars imgpars(image_size, cell_size, padding_factor, stp::KernelFunction::PSWF, support, kernel_exact, oversampling, gen_beam, gridding_correction, analytic_gcf);
        std::pair<arma::Mat<real_t>, arma::Mat<real_t>> orig_result = image_visibilities(PSWF(trunc), vis, vis_weights, uvw_lambda, imgpars);

#ifndef FFTSHIFT
        // Output matrices need to be shifted because image_visibilities does not shift them
        fftshift(orig_result.first);
        fftshift(orig_result.second);
#endif

        result.first = arma::conv_to<arma::mat>::from(orig_result.first);
        result.second = arma::conv_to<arma::mat>::from(orig_result.second);
    }
};

TEST(ImagerPSWF, SmallImage)
{
    imager_test_pswf pswf_small_image("pswf", "small_image");
    EXPECT_NEAR(arma::accu(pswf_small_image.result.first), arma::accu(arma::real(pswf_small_image.expected_result.first)), fptolerance);
    EXPECT_TRUE(arma::approx_equal(pswf_small_image.result.second, arma::real(pswf_small_image.expected_result.second), "absdiff", fptolerance));
}

TEST(ImagerPSWF, MediumImage)
{
    imager_test_pswf pswf_medium_image("pswf", "medium_image");
    EXPECT_TRUE(arma::approx_equal(pswf_medium_image.result.first, arma::real(pswf_medium_image.expected_result.first), "absdiff", fptolerance));
    EXPECT_TRUE(arma::approx_equal(pswf_medium_image.result.second, arma::real(pswf_medium_image.expected_result.second), "absdiff", fptolerance));
}

TEST(ImagerPSWF, LargeImage)
{
    imager_test_pswf pswf_large_image("pswf", "large_image");
    EXPECT_TRUE(arma::approx_equal(pswf_large_image.result.first, arma::real(pswf_large_image.expected_result.first), "absdiff", fptolerance));
    EXPECT_TRUE(arma::approx_equal(pswf_large_image.result.second, arma::real(pswf_large_image.expected_result.second), "absdiff", fptolerance));
}
