/** @file pipeline_func_Triangle.cpp
 *  @brief Test Pipeline with Triangle
 *
 *  TestCase to test the pipeline-imager
 *  implementation with triangle convolution
 *
 *  @bug No known bugs.
 */

#include "pipeline_func.h"

struct pipeline_test_triangle : public PipelineHandler {
    double half_base_width;
    double triangle_value;

public:
    pipeline_test_triangle(const std::string typeConvolution, const std::string typeTest)
        : PipelineHandler(typeConvolution, typeTest)
    {
        half_base_width = val["half_base_width"].GetDouble();
        triangle_value = val["triangle_value"].GetDouble();

        cnpy::NpyArray vis_npy(cnpy::npz_load(val["input_file"].GetString(), "vis"));
        cnpy::NpyArray uvw_npy(cnpy::npz_load(val["input_file"].GetString(), "uvw"));
        std::string expected_results_path = val["expected_results"].GetString();

        arma::cx_mat vis(load_npy_array(vis_npy));
        arma::cx_mat uvw_lambda(load_npy_array(uvw_npy));

        cnpy::NpyArray image_array(cnpy::npz_load(expected_results_path, "image"));
        cnpy::NpyArray beam_array(cnpy::npz_load(expected_results_path, "beam"));

        // Loads the expected results to a cube
        expected_result = arma::cx_cube(image_size, image_size, 2);
        expected_result.slice(0) = load_npy_array(image_array);
        expected_result.slice(1) = load_npy_array(beam_array);

        result = image_visibilities(vis, uvw_lambda, image_size, cell_size, support, oversampling, pad, normalize, Triangle(half_base_width, triangle_value));
    }
};

// Benchmark functions
void BM_pipeline_test_triangle(benchmark::State& state)
{

    while (state.KeepRunning()) {
        switch (state.range(0)) {
        case 0: {
            pipeline_test_triangle triangle("triangle", "small_image");
            break;
        }
        case 1: {
            pipeline_test_triangle triangle("triangle", "medium_image");
            break;
        }
        case 2: {
            pipeline_test_triangle triangle("triangle", "large_image");
            break;
        }
        }
    }
}

TEST(PipelineTriangle, SmallImage)
{
    pipeline_test_triangle triangle("triangle", "small_image");
    EXPECT_TRUE(arma::approx_equal(triangle.result, triangle.expected_result, "absdiff", tolerance));
}

TEST(PipelineTriangle, MediumImage)
{
    pipeline_test_triangle triangle("triangle", "medium_image");
    EXPECT_TRUE(arma::approx_equal(triangle.result, triangle.expected_result, "absdiff", tolerance));
}

TEST(PipelineTriangle, LargeImage)
{
    pipeline_test_triangle triangle("triangle", "large_image");
    EXPECT_TRUE(arma::approx_equal(triangle.result, triangle.expected_result, "absdiff", tolerance));
}

TEST(PipelineTriangle, PipelineTriangle_benchmark)
{

    BENCHMARK(BM_pipeline_test_triangle)
        ->Range(0, 2);

    benchmark::RunSpecifiedBenchmarks();
}
