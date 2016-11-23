/** @file pipeline_func_Triangle.cpp
 *  @brief Test Pipeline with Triangle
 *
 *  TestCase to test the pipeline-imager
 *  implementation with triangle convolution
 *
 *  @bug No known bugs.
 */

#include "pipeline_func.h"

struct TrianglePipeline : public PipelineHandler {
    double half_base_width;
    double triangle_value;

    void set_up_results();

public:
    TrianglePipeline(const char* typeConvolution, const char* typeTest)
        : PipelineHandler(typeConvolution, typeTest)
    {
        set_up_results();
    }
};

void TrianglePipeline::set_up_results()
{
    half_base_width = d["half_base_width"].GetDouble();
    triangle_value = d["triangle_value"].GetDouble();

    cnpy::NpyArray vis1(cnpy::npz_load(d["input_file"].GetString(), "vis"));
    cnpy::NpyArray uvw1(cnpy::npz_load(d["input_file"].GetString(), "uvw"));
    std::string expected_results_path = d["expected_results"].GetString();

    arma::cx_mat vis(load_npy_array(&vis1));
    arma::cx_mat uvw_lambda(load_npy_array(&uvw1));

    Triangle triangle(half_base_width, triangle_value);

    result = image_visibilities(vis, uvw_lambda, image_size, cell_size, support, oversampling, pad, normalize, triangle);

    cnpy::NpyArray image_array(cnpy::npz_load(expected_results_path, "image"));
    cnpy::NpyArray beam_array(cnpy::npz_load(expected_results_path, "beam"));

    // Loads the expected results to a cube
    expected_result = arma::cx_cube(image_size, image_size, 2);
    expected_result.slice(0) = load_npy_array(&image_array);
    expected_result.slice(1) = load_npy_array(&beam_array);
}

TEST(PipelineTriangle, Small_Image)
{
    TrianglePipeline triangle("triangle", "small_image");
    EXPECT_TRUE(arma::approx_equal(triangle.result, triangle.expected_result, "absdiff", tolerance));
}

TEST(PipelineTriangle, Medium_Image)
{
    TrianglePipeline triangle("triangle", "medium_image");
    EXPECT_TRUE(arma::approx_equal(triangle.result, triangle.expected_result, "absdiff", tolerance));
}

TEST(PipelineTriangle, Large_Image)
{
    TrianglePipeline triangle("triangle", "large_image");
    EXPECT_TRUE(arma::approx_equal(triangle.result, triangle.expected_result, "absdiff", tolerance));
}
