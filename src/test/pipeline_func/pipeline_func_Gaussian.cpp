/** @file pipeline_func_Gaussian.cpp
 *  @brief Test Pipeline with Gaussian
 *
 *  TestCase to test the pipeline-imager
 *  implementation with gaussian convolution
 *
 *  @bug No known bugs.
 */

#include "pipeline_func.h"

struct GaussianPipeline : public PipelineHandler {
    double width_normalization;
    double threshold;

    void set_up_results();

public:
    GaussianPipeline(const char* typeConvolution, const char* typeTest)
        : PipelineHandler(typeConvolution, typeTest)
    {
        set_up_results();
    }
};

void GaussianPipeline::set_up_results()
{
    width_normalization = d["width_normalization"].GetDouble();
    threshold = d["threshold"].GetDouble();
    std::string expected_results_path = d["expected_results"].GetString();

    cnpy::NpyArray vis1(cnpy::npz_load(d["input_file"].GetString(), "vis"));
    cnpy::NpyArray uvw1(cnpy::npz_load(d["input_file"].GetString(), "uvw"));

    arma::cx_mat vis(load_npy_array(&vis1));
    arma::cx_mat uvw_lambda(load_npy_array(&uvw1));

    Gaussian gaussian(width_normalization, threshold);

    result = image_visibilities(vis, uvw_lambda, image_size, cell_size, support, oversampling, pad, normalize, gaussian);

    cnpy::NpyArray image_array(cnpy::npz_load(expected_results_path, "image"));
    cnpy::NpyArray beam_array(cnpy::npz_load(expected_results_path, "beam"));

    // Loads the expected results to a cube
    expected_result = arma::cx_cube(image_size, image_size, 2);
    expected_result.slice(0) = load_npy_array(&image_array);
    expected_result.slice(1) = load_npy_array(&beam_array);
}

TEST(PipelineGaussian, Small_Image)
{
    GaussianPipeline gaussian("gaussian", "small_image");
    EXPECT_TRUE(arma::approx_equal(gaussian.result, gaussian.expected_result, "absdiff", tolerance));
}

TEST(PipelineGaussian, Medium_Image)
{
    GaussianPipeline gaussian("gaussian", "medium_image");
    EXPECT_TRUE(arma::approx_equal(gaussian.result, gaussian.expected_result, "absdiff", tolerance));
}

TEST(PipelineGaussian, Large_Image)
{
    GaussianPipeline gaussian("gaussian", "large_image");
    EXPECT_TRUE(arma::approx_equal(gaussian.result, gaussian.expected_result, "absdiff", tolerance));
}
