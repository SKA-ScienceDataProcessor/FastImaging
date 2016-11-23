/** @file pipeline_func_GaussianSinc.cpp
 *  @brief Test Pipeline with GaussianSinc
 *
 *  TestCase to test the pipeline-imager
 *  implementation with gaussiansinc convolution
 *
 *  @bug No known bugs.
 */

#include "pipeline_func.h"

struct GaussianSincPipeline : public PipelineHandler {
    double width_normalization_gaussian;
    double width_normalization_sinc;
    double trunc;

    void set_up_results();

public:
    GaussianSincPipeline(const char* typeConvolution, const char* typeTest)
        : PipelineHandler(typeConvolution, typeTest)
    {
        set_up_results();
    }
};

void GaussianSincPipeline::set_up_results()
{
    width_normalization_gaussian = d["width_normalization_gaussian"].GetDouble();
    width_normalization_sinc = d["width_normalization_sinc"].GetDouble();
    trunc = d["trunc"].GetDouble();
    std::string expected_results_path = d["expected_results"].GetString();

    cnpy::NpyArray vis1(cnpy::npz_load(d["input_file"].GetString(), "vis"));
    cnpy::NpyArray uvw1(cnpy::npz_load(d["input_file"].GetString(), "uvw"));

    arma::cx_mat vis(load_npy_array(&vis1));
    arma::cx_mat uvw_lambda(load_npy_array(&uvw1));

    GaussianSinc gaussiansinc(width_normalization_gaussian, width_normalization_sinc, trunc);

    result = image_visibilities(vis, uvw_lambda, image_size, cell_size, support, oversampling, pad, normalize, gaussiansinc);

    cnpy::NpyArray image_array(cnpy::npz_load(expected_results_path, "image"));
    cnpy::NpyArray beam_array(cnpy::npz_load(expected_results_path, "beam"));

    // Loads the expected results to a cube
    expected_result = arma::cx_cube(image_size, image_size, 2);
    expected_result.slice(0) = load_npy_array(&image_array);
    expected_result.slice(1) = load_npy_array(&beam_array);
}

TEST(PipelineGaussianSinc, Small_Image)
{
    GaussianSincPipeline gaussiansinc("gaussiansinc", "small_image");
    EXPECT_TRUE(arma::approx_equal(gaussiansinc.result, gaussiansinc.expected_result, "absdiff", tolerance));
}

TEST(PipelineGaussianSinc, Medium_Image)
{
    GaussianSincPipeline gaussiansinc("gaussiansinc", "medium_image");
    EXPECT_TRUE(arma::approx_equal(gaussiansinc.result, gaussiansinc.expected_result, "absdiff", tolerance));
}

TEST(PipelineGaussianSinc, Large_Image)
{
    GaussianSincPipeline gaussiansinc("gaussiansinc", "large_image");
    EXPECT_TRUE(arma::approx_equal(gaussiansinc.result, gaussiansinc.expected_result, "absdiff", tolerance));
}
