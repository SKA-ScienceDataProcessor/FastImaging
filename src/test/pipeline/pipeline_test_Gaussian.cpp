/** @file pipeline_func_Gaussian.cpp
 *  @brief Test Pipeline with Gaussian
 *
 *  TestCase to test the pipeline-imager
 *  implementation with gaussian convolution
 *
 *  @bug No known bugs.
 */

#include "pipeline_func.h"

struct pipeline_test_gaussian : public PipelineHandler {
    double width_normalization;
    double threshold;

public:
    pipeline_test_gaussian(const std::string typeConvolution, const std::string typeTest)
        : PipelineHandler(typeConvolution, typeTest)
    {
        width_normalization = val["width_normalization"].GetDouble();
        threshold = val["threshold"].GetDouble();
        std::string expected_results_path = val["expected_results"].GetString();

        cnpy::NpyArray vis_npy(cnpy::npz_load(val["input_file"].GetString(), "vis"));
        cnpy::NpyArray uvw_npy(cnpy::npz_load(val["input_file"].GetString(), "uvw"));

        arma::cx_mat vis(load_npy_array(vis_npy));
        arma::cx_mat uvw_lambda(load_npy_array(uvw_npy));

        cnpy::NpyArray image_array(cnpy::npz_load(expected_results_path, "image"));
        cnpy::NpyArray beam_array(cnpy::npz_load(expected_results_path, "beam"));

        // Loads the expected results to a cube
        expected_result = arma::cx_cube(image_size, image_size, 2);
        expected_result.slice(0) = load_npy_array(image_array);
        expected_result.slice(1) = load_npy_array(beam_array);

        result = image_visibilities(vis, uvw_lambda, image_size, cell_size, support, oversampling, pad, normalize, Gaussian(width_normalization, threshold));
    }
};

// Benchmark functions
void BM_pipeline_test_gaussian(benchmark::State& state)
{

    while (state.KeepRunning()) {
        switch (state.range(0)) {
        case 0: {
            pipeline_test_gaussian gaussian("gaussian", "small_image");
            break;
        }
        case 1: {
            pipeline_test_gaussian gaussian("gaussian", "medium_image");
            break;
        }
        case 2: {
            pipeline_test_gaussian gaussian("gaussian", "large_image");
            break;
        }
        }
    }
}

TEST(PipelineGaussian, SmallImage)
{
    pipeline_test_gaussian gaussian("gaussian", "small_image");
    EXPECT_TRUE(arma::approx_equal(gaussian.result, gaussian.expected_result, "absdiff", tolerance));
}

TEST(PipelineGaussian, MediumImage)
{
    pipeline_test_gaussian gaussian("gaussian", "medium_image");
    EXPECT_TRUE(arma::approx_equal(gaussian.result, gaussian.expected_result, "absdiff", tolerance));
}

TEST(PipelineGaussian, LargeImage)
{
    pipeline_test_gaussian gaussian("gaussian", "large_image");
    EXPECT_TRUE(arma::approx_equal(gaussian.result, gaussian.expected_result, "absdiff", tolerance));
}

TEST(PipelineGaussian, PipelineGaussian_benchmark)
{

    BENCHMARK(BM_pipeline_test_gaussian)
        ->Range(0, 2);

    benchmark::RunSpecifiedBenchmarks();
}
