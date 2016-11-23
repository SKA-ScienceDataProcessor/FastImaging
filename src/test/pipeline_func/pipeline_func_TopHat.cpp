/** @file pipeline_func_TopHat.cpp
 *  @brief Test Pipeline with Tophat
 *
 *  TestCase to test the pipeline-imager
 *  implementation with tophat convolution
 *
 *  @bug No known bugs.
 */

#include "pipeline_func.h"

struct TopHatPipeline : public PipelineHandler {
    double half_base_width;

    void set_up_results();

public:
    TopHatPipeline(const char* typeConvolution, const char* typeTest)
        : PipelineHandler(typeConvolution, typeTest)
    {
        set_up_results();
    }
};

void TopHatPipeline::set_up_results()
{
    half_base_width = d["half_base_width"].GetDouble();

    cnpy::NpyArray vis1(cnpy::npz_load(d["input_file"].GetString(), "vis"));
    cnpy::NpyArray uvw1(cnpy::npz_load(d["input_file"].GetString(), "uvw"));
    std::string expected_results_path = d["expected_results"].GetString();

    arma::cx_mat vis(load_npy_array(&vis1));
    arma::cx_mat uvw_lambda(load_npy_array(&uvw1));

    TopHat tophat(half_base_width);

    result = image_visibilities(vis, uvw_lambda, image_size, cell_size, support, oversampling, pad, normalize, tophat);

    cnpy::NpyArray image_array(cnpy::npz_load(expected_results_path, "image"));
    cnpy::NpyArray beam_array(cnpy::npz_load(expected_results_path, "beam"));

    // Loads the expected results to a cube
    expected_result = arma::cx_cube(image_size, image_size, 2);
    expected_result.slice(0) = load_npy_array(&image_array);
    expected_result.slice(1) = load_npy_array(&beam_array);
}

TEST(PipelineTopHat, Small_Image)
{
    TopHatPipeline tophat("tophat", "small_image");
    EXPECT_TRUE(arma::approx_equal(tophat.result, tophat.expected_result, "absdiff", tolerance));
}

TEST(PipelineTopHat, Medium_Image)
{
    TopHatPipeline tophat("tophat", "medium_image");
    EXPECT_TRUE(arma::approx_equal(tophat.result, tophat.expected_result, "absdiff", tolerance));
}

TEST(PipelineTopHat, Large_Image)
{
    TopHatPipeline tophat("tophat", "large_image");
    EXPECT_TRUE(arma::approx_equal(tophat.result, tophat.expected_result, "absdiff", tolerance));
}
