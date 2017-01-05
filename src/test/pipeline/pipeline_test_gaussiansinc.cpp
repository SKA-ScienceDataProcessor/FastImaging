#include "../auxiliary/load_data.h"
#include <gtest/gtest.h>
#include <stp.h>

// Cmake variable
#ifndef _TESTPATH
#define _TESTPATH 0
#endif

using namespace stp;

TEST(PipelineGaussianSincFunc, test_gaussian_sinc)
{
    const double width_normalization_gaussian(2.52);
    const double width_normalization_sinc(1.55);
    const double trunc(3.0);
    const double kernel_support = 3;
    const double vis_noise_level(0.001);
    const double image_size(1024);
    const double cell_size(3);
    const double detection_n_sigma(50);
    const double analysis_n_sigma(25);

    std::string path_file;
    if (_TESTPATH) {
        path_file = _TESTPATH;
    }

    cnpy::NpyArray uvw_npy(cnpy::npz_load(path_file + "mock_uvw_vis.npz", "uvw"));
    arma::mat uvw_lambda = load_npy_double_array(uvw_npy);

    source_find_image pipeline = generate_pipeline(
        uvw_lambda,
        vis_noise_level,
        image_size,
        cell_size,
        detection_n_sigma,
        analysis_n_sigma,
        kernel_support,
        GaussianSinc(width_normalization_gaussian, width_normalization_sinc, trunc));

    int total_islands = pipeline.islands.size();

    EXPECT_EQ(total_islands, 0);
}
