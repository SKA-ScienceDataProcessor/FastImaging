/** @file pipeline_func_Sinc.cpp
 *  @brief Test Pipeline with Sinc
 *
 *  TestCase to test the pipeline-imager
 *  implementation with sinc convolution
 *
 *  @bug No known bugs.
 */

#include "pipeline_func.h"

struct SincPipeline : public PipelineHandler {
    double width_normalization;
    double threshold;

    void set_up_results();

public:
    SincPipeline(const char* typeConvolution, const char* typeTest)
        : PipelineHandler(typeConvolution, typeTest)
    {
        set_up_results();
    }
};

void SincPipeline::set_up_results()
{
    width_normalization = d["width_normalization"].GetDouble();
    threshold = d["threshold"].GetDouble();
    std::string expected_results_path = d["expected_results"].GetString();

    cnpy::NpyArray vis1(cnpy::npz_load(d["input_file"].GetString(), "vis"));
    cnpy::NpyArray uvw1(cnpy::npz_load(d["input_file"].GetString(), "uvw"));

    arma::cx_mat vis(load_npy_array(&vis1));
    arma::cx_mat uvw_lambda(load_npy_array(&uvw1));

    Sinc sinc(width_normalization, threshold);

    result = image_visibilities(vis, uvw_lambda, image_size, cell_size, support, oversampling, pad, normalize, sinc);

    cnpy::NpyArray image_array(cnpy::npz_load(expected_results_path, "image"));
    cnpy::NpyArray beam_array(cnpy::npz_load(expected_results_path, "beam"));

    // Loads the expected results to a cube
    expected_result = arma::cx_cube(image_size, image_size, 2);
    expected_result.slice(0) = load_npy_array(&image_array);
    expected_result.slice(1) = load_npy_array(&beam_array);
}

TEST(PipelineSinc, Small_Image)
{
    SincPipeline sinc("sinc", "small_image");
    EXPECT_TRUE(arma::approx_equal(sinc.result, sinc.expected_result, "absdiff", tolerance));
}

TEST(PipelineSinc, Medium_Image)
{
    SincPipeline sinc("sinc", "medium_image");
    EXPECT_TRUE(arma::approx_equal(sinc.result, sinc.expected_result, "absdiff", tolerance));
}

TEST(PipelineSinc, Large_Image)
{
    SincPipeline sinc("sinc", "large_image");
    EXPECT_TRUE(arma::approx_equal(sinc.result, sinc.expected_result, "absdiff", tolerance));
}
