/** @file gridder_test_benchmark.cpp
 *  @brief Test Gridder module performance
 *
 *  @bug No known bugs.
 */
#include <armadillo>
#include <benchmark/benchmark.h>
#include <load_data.h>
#include <load_json_config.h>
#include <stp.h>

// Cmake variable
#ifndef _PIPELINE_TESTPATH
#define _PIPELINE_TESTPATH 0
#endif

std::string data_path(_PIPELINE_DATAPATH);
std::string input_npz("simdata_small.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config.json");
std::string config_file_oversampling("fastimg_oversampling_config.json");

stp::source_find_image run_pipeline(arma::mat uvw_lambda, arma::cx_mat model_vis, arma::cx_mat data_vis, int image_size, double cell_size, double detection_n_sigma, double analysis_n_sigma, int support = 3, bool kernel_exact = true, int oversampling = 1)
{
    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = data_vis - model_vis;

    stp::GaussianSinc kernel_func(support);
    std::pair<arma::cx_mat, arma::cx_mat> result = stp::image_visibilities(kernel_func, residual_vis, uvw_lambda, image_size, cell_size, support, kernel_exact, oversampling);

    return stp::source_find_image(arma::real(result.first), detection_n_sigma, analysis_n_sigma, 0.0, true);
}

static void pipeline_kernel_exact_benchmark(benchmark::State& state)
{
    int image_size = pow(2, state.range(0));

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_model = load_npy_complex_array(data_path + input_npz, "model");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");
    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_exact);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(run_pipeline(input_uvw, input_model, input_vis, image_size, cfg.cell_size, cfg.detection_n_sigma, cfg.analysis_n_sigma, cfg.kernel_support, cfg.kernel_exact));
    }
}

static void pipeline_kernel_oversampling_benchmark(benchmark::State& state)
{
    int image_size = pow(2, state.range(0));

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_model = load_npy_complex_array(data_path + input_npz, "model");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");
    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(run_pipeline(input_uvw, input_model, input_vis, image_size, cfg.cell_size, cfg.detection_n_sigma, cfg.analysis_n_sigma, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling));
    }
}

BENCHMARK(pipeline_kernel_oversampling_benchmark)
    ->Args({ 10 })
    ->Args({ 11 })
    ->Args({ 12 })
    ->Args({ 13 })
    ->Args({ 14 })
    ->Unit(benchmark::kMillisecond);

BENCHMARK(pipeline_kernel_exact_benchmark)
    ->Args({ 10 })
    ->Args({ 11 })
    ->Args({ 12 })
    ->Args({ 13 })
    ->Args({ 14 })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN()
