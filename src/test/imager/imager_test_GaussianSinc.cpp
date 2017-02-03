/** @file imager_func_GaussianSinc.cpp
 *  @brief Test imager with GaussianSinc
 *
 *  TestCase to test the imager-imager
 *  implementation with gaussiansinc convolution
 *
 *  @bug No known bugs.
 */

#include "load_json_imager.h"
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

struct imager_test_gaussiansinc : public ImagerHandler {
    double width_normalization_gaussian;
    double width_normalization_sinc;
    double trunc;

public:
    imager_test_gaussiansinc() = default;
    imager_test_gaussiansinc(const std::string& typeConvolution, const std::string& typeTest)
        : ImagerHandler(typeConvolution, typeTest)
    {
        width_normalization_gaussian = val["width_normalization_gaussian"].GetDouble();
        width_normalization_sinc = val["width_normalization_sinc"].GetDouble();
        trunc = val["trunc"].GetDouble();
        std::string expected_results_path = val["expected_results"].GetString();

        arma::cx_mat vis(load_npy_complex_array(val["input_file"].GetString(), "vis"));
        arma::mat uvw_lambda(load_npy_double_array(val["input_file"].GetString(), "uvw"));

        // Loads the expected results to a arma::mat pair
        expected_result = std::make_pair(std::move(load_npy_complex_array(expected_results_path, "image")), std::move(load_npy_complex_array(expected_results_path, "beam")));

        result = image_visibilities(GaussianSinc(width_normalization_gaussian, width_normalization_sinc, trunc), vis, uvw_lambda, image_size, cell_size, support, kernel_exact, oversampling);
    }
};

TEST(ImagerGaussianSinc, SmallImage)
{
    imager_test_gaussiansinc gaussiansinc_small_image("gaussiansinc", "small_image");
    EXPECT_TRUE(arma::approx_equal(gaussiansinc_small_image.result.first, gaussiansinc_small_image.expected_result.first, "absdiff", tolerance));
    EXPECT_TRUE(arma::approx_equal(gaussiansinc_small_image.result.second, gaussiansinc_small_image.expected_result.second, "absdiff", tolerance));
}

TEST(ImagerGaussianSinc, MediumImage)
{
    imager_test_gaussiansinc gaussiansinc_medium_image("gaussiansinc", "medium_image");
    EXPECT_TRUE(arma::approx_equal(gaussiansinc_medium_image.result.first, gaussiansinc_medium_image.expected_result.first, "absdiff", tolerance));
    EXPECT_TRUE(arma::approx_equal(gaussiansinc_medium_image.result.second, gaussiansinc_medium_image.expected_result.second, "absdiff", tolerance));
}

TEST(ImagerGaussianSinc, LargeImage)
{
    imager_test_gaussiansinc gaussiansinc_large_image("gaussiansinc", "large_image");
    EXPECT_TRUE(arma::approx_equal(gaussiansinc_large_image.result.first, gaussiansinc_large_image.expected_result.first, "absdiff", tolerance));
    EXPECT_TRUE(arma::approx_equal(gaussiansinc_large_image.result.second, gaussiansinc_large_image.expected_result.second, "absdiff", tolerance));
}
