#include <gtest/gtest.h>
#include <load_data.h>
#include <load_json_config.h>
#include <stp.h>

using namespace stp;

std::string data_path(_PIPELINE_DATAPATH);
std::string input_npz("simdata_small.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config1.json");
std::string config_file_oversampling("fastimg_oversampling_config1.json");

const double dtolerance(1.0e-10);

stp::source_find_image run_pipeline(arma::mat uvw_lambda, arma::cx_mat model_vis, arma::cx_mat data_vis, int image_size, double cell_size, double detection_n_sigma, double analysis_n_sigma, int kernel_support = 3, bool kernel_exact = true, int oversampling = 1)
{
    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = data_vis - model_vis;

    stp::GaussianSinc kernel_func(kernel_support);
    std::pair<arma::cx_mat, arma::cx_mat> result = stp::image_visibilities(kernel_func, residual_vis, uvw_lambda, image_size, cell_size, kernel_support, kernel_exact, oversampling);

    return stp::source_find_image(arma::real(result.first), detection_n_sigma, analysis_n_sigma, 0.0, true);
}

TEST(PipelineGaussianSincExact, test_gaussian_sinc_exact)
{
    //Load simulated data from input_npz
    arma::mat input_uvw;
    arma::cx_mat input_model, input_vis;
    input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    input_model = load_npy_complex_array(data_path + input_npz, "model");
    input_vis = load_npy_complex_array(data_path + input_npz, "vis");
    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_exact);

    stp::source_find_image sfimage = run_pipeline(input_uvw, input_model, input_vis, cfg.image_size, cfg.cell_size, cfg.detection_n_sigma, cfg.analysis_n_sigma, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling);

    int total_islands = sfimage.islands.size();
    int expected_total_islands = 1;
    EXPECT_EQ(total_islands, expected_total_islands);

    int label_idx = sfimage.islands[0].label_idx;
    int sign = sfimage.islands[0].sign;
    double extremum_val = sfimage.islands[0].extremum_val;
    int extremum_x_idx = sfimage.islands[0].extremum_x_idx;
    int extremum_y_idx = sfimage.islands[0].extremum_y_idx;
    double xbar = sfimage.islands[0].xbar;
    double ybar = sfimage.islands[0].ybar;

    int expected_label_idx = 1;
    int expected_sign = 1;
    double expected_extremum_val = 0.11044656115547172;
    int expected_extremum_x_idx = 824;
    int expected_extremum_y_idx = 872;
    double expected_xbar = 823.48705108625256;
    double expected_ybar = 871.63927768390715;

    EXPECT_EQ(label_idx, expected_label_idx);
    EXPECT_EQ(sign, expected_sign);
    EXPECT_DOUBLE_EQ(extremum_val, expected_extremum_val);
    EXPECT_EQ(extremum_x_idx, expected_extremum_x_idx);
    EXPECT_EQ(extremum_y_idx, expected_extremum_y_idx);
    EXPECT_DOUBLE_EQ(xbar, expected_xbar);
    EXPECT_DOUBLE_EQ(ybar, expected_ybar);
}

TEST(PipelineGaussianSincOversampling, test_gaussian_sinc_oversampling)
{
    //Load simulated data from input_npz
    arma::mat input_uvw;
    arma::cx_mat input_model, input_vis;
    input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    input_model = load_npy_complex_array(data_path + input_npz, "model");
    input_vis = load_npy_complex_array(data_path + input_npz, "vis");
    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    stp::source_find_image sfimage = run_pipeline(input_uvw, input_model, input_vis, cfg.image_size, cfg.cell_size, cfg.detection_n_sigma, cfg.analysis_n_sigma, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling);

    int total_islands = sfimage.islands.size();
    int expected_total_islands = 1;
    EXPECT_EQ(total_islands, expected_total_islands);

    int label_idx = sfimage.islands[0].label_idx;
    int sign = sfimage.islands[0].sign;
    double extremum_val = sfimage.islands[0].extremum_val;
    int extremum_x_idx = sfimage.islands[0].extremum_x_idx;
    int extremum_y_idx = sfimage.islands[0].extremum_y_idx;
    double xbar = sfimage.islands[0].xbar;
    double ybar = sfimage.islands[0].ybar;

    int expected_label_idx = 1;
    int expected_sign = 1;
    double expected_extremum_val = 0.11010496366579756;
    int expected_extremum_x_idx = 824;
    int expected_extremum_y_idx = 872;
    double expected_xbar = 823.48710510938361;
    double expected_ybar = 871.63972539949111;

    EXPECT_EQ(label_idx, expected_label_idx);
    EXPECT_EQ(sign, expected_sign);
    EXPECT_TRUE(std::abs(extremum_val - expected_extremum_val) < dtolerance);
    EXPECT_EQ(extremum_x_idx, expected_extremum_x_idx);
    EXPECT_EQ(extremum_y_idx, expected_extremum_y_idx);
    EXPECT_TRUE(std::abs(xbar - expected_xbar) < dtolerance);
    EXPECT_TRUE(std::abs(ybar - expected_ybar) < dtolerance);
}
