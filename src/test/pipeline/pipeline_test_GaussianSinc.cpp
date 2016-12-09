/** @file pipeline_func_GaussianSinc.cpp
 *  @brief Test Pipeline with GaussianSinc
 *
 *  TestCase to test the pipeline-imager
 *  implementation with gaussiansinc convolution
 *
 *  @bug No known bugs.
 */

#include "pipeline_func.h"

struct pipeline_test_gaussiansinc : public PipelineHandler {
    double width_normalization_gaussian;
    double width_normalization_sinc;
    double trunc;

public:
    pipeline_test_gaussiansinc(const std::string typeConvolution, const std::string typeTest)
        : PipelineHandler(typeConvolution, typeTest)
    {
        width_normalization_gaussian = val["width_normalization_gaussian"].GetDouble();
        width_normalization_sinc = val["width_normalization_sinc"].GetDouble();
        trunc = val["trunc"].GetDouble();
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

        result = image_visibilities(vis, uvw_lambda, image_size, cell_size, support, oversampling, pad, normalize, GaussianSinc(width_normalization_gaussian, width_normalization_sinc, trunc));
    }
};

// Benchmark functions
void BM_pipeline_test_gaussiansinc(benchmark::State& state)
{

    while (state.KeepRunning()) {
        switch (state.range(0)) {
        case 0: {
            pipeline_test_gaussiansinc gaussiansinc("gaussiansinc", "small_image");
            break;
        }
        case 1: {
            pipeline_test_gaussiansinc gaussiansinc("gaussiansinc", "medium_image");
            break;
        }
        case 2: {
            pipeline_test_gaussiansinc gaussiansinc("gaussiansinc", "large_image");
            break;
        }
        }
    }
}

TEST(PipelineGaussianSinc, SmallImage)
{
    pipeline_test_gaussiansinc gaussiansinc("gaussiansinc", "small_image");
    EXPECT_TRUE(arma::approx_equal(gaussiansinc.result, gaussiansinc.expected_result, "absdiff", tolerance));
}

TEST(PipelineGaussianSinc, MediumImage)
{
    pipeline_test_gaussiansinc gaussiansinc("gaussiansinc", "medium_image");
    EXPECT_TRUE(arma::approx_equal(gaussiansinc.result, gaussiansinc.expected_result, "absdiff", tolerance));
}

TEST(PipelineGaussianSinc, LargeImage)
{
    pipeline_test_gaussiansinc gaussiansinc("gaussiansinc", "large_image");
    EXPECT_TRUE(arma::approx_equal(gaussiansinc.result, gaussiansinc.expected_result, "absdiff", tolerance));
}

TEST(PipelineGaussianSinc, PipelineGaussianSinc_benchmark)
{

    BENCHMARK(BM_pipeline_test_gaussiansinc)
        ->Range(0, 0);

    benchmark::RunSpecifiedBenchmarks();
}
