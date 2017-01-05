/** @file imager_func_Gaussian.cpp
 *  @brief Test imager with Gaussian
 *
 *  TestCase to test the imager-imager
 *  implementation with gaussian convolution
 *
 *  @bug No known bugs.
 */

#include "../auxiliary/load_json_imager.h"
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

        cnpy::NpyArray vis_npy(cnpy::npz_load(val["input_file"].GetString(), "vis"));
        cnpy::NpyArray uvw_npy(cnpy::npz_load(val["input_file"].GetString(), "uvw"));

        arma::cx_mat vis(load_npy_complex_array(vis_npy));
        arma::mat uvw_lambda(load_npy_double_array(uvw_npy));

        cnpy::NpyArray image_array(cnpy::npz_load(expected_results_path, "image"));
        cnpy::NpyArray beam_array(cnpy::npz_load(expected_results_path, "beam"));

        // Loads the expected results to a arma::mat pair
        expected_result = std::make_pair(std::move(load_npy_complex_array(image_array)), std::move(load_npy_complex_array(beam_array)));

        std::experimental::optional<int> oversampling;
        if (use_oversampling == true) {
            oversampling = oversampling_val;
        }
        result = image_visibilities(Gaussian(width_normalization, threshold), vis, uvw_lambda, image_size, cell_size, support, oversampling);
    }
};

TEST(ImagerGaussian, SmallImage)
{
    imager_test_gaussian gaussian_small_image("gaussian", "small_image");
    EXPECT_TRUE(arma::approx_equal(gaussian_small_image.result.first, gaussian_small_image.expected_result.first, "absdiff", tolerance));
    EXPECT_TRUE(arma::approx_equal(gaussian_small_image.result.second, gaussian_small_image.expected_result.second, "absdiff", tolerance));
}

TEST(ImagerGaussian, MediumImage)
{
    imager_test_gaussian gaussian_medium_image("gaussian", "medium_image");
    EXPECT_TRUE(arma::approx_equal(gaussian_medium_image.result.first, gaussian_medium_image.expected_result.first, "absdiff", tolerance));
    EXPECT_TRUE(arma::approx_equal(gaussian_medium_image.result.second, gaussian_medium_image.expected_result.second, "absdiff", tolerance));
}

TEST(ImagerGaussian, LargeImage)
{
    imager_test_gaussian gaussian_large_image("gaussian", "large_image");
    EXPECT_TRUE(arma::approx_equal(gaussian_large_image.result.first, gaussian_large_image.expected_result.first, "absdiff", tolerance));
    EXPECT_TRUE(arma::approx_equal(gaussian_large_image.result.second, gaussian_large_image.expected_result.second, "absdiff", tolerance));
}
