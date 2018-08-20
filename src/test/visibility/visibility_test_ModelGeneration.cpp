#include "../../auxiliary/load_data.h"
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

std::string data_path(_VISIBILITY_TESTPATH);
std::string input_vis_npz("simple_vis.npz");
std::string model_vis_npz("expected_model.npz");

// Test generation of model visibilities
TEST(ModelVisibilityGeneration, test_visibility)
{
    const double vis_tolerance = 10e-12;

    // Load UVW-baselines from input_npz
    arma::mat input_uvw = load_npy_double_array<double>(data_path + input_vis_npz, "uvw_lambda");
    // Load skymodel data
    arma::mat skymodel = load_npy_double_array<double>(data_path + input_vis_npz, "skymodel");
    // Load expected model visibilities
    arma::cx_mat expected_model_vis = load_npy_complex_array<double>(data_path + model_vis_npz, "model");

    // Generate model visibilities from the skymodel and UVW-baselines
    arma::cx_mat model_vis = stp::generate_visibilities_from_local_skymodel(skymodel, input_uvw);

    // Check if generated model visibilities correspond to the expected ones
    EXPECT_TRUE(arma::approx_equal(expected_model_vis, model_vis, "absdiff", vis_tolerance));
}
