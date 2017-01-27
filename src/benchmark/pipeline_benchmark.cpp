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
std::string input_npz("simdata.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config2.json");
std::string config_file_oversampling("fastimg_oversampling_config2.json");

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
    //Load simulated data from input_npz
    arma::mat input_uvw;
    arma::cx_mat input_model, input_vis;
    load_npz_simdata(data_path + input_npz, input_uvw, input_model, input_vis);
    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_exact);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(run_pipeline(input_uvw, input_model, input_vis, state.range(0), cfg.cell_size, cfg.detection_n_sigma, cfg.analysis_n_sigma, cfg.kernel_support, cfg.kernel_exact));
    }
}

static void pipeline_kernel_oversampling_benchmark(benchmark::State& state)
{
    //Load simulated data from input_npz
    arma::mat input_uvw;
    arma::cx_mat input_model, input_vis;
    load_npz_simdata(data_path + input_npz, input_uvw, input_model, input_vis);
    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    while (state.KeepRunning()) {
        stp::source_find_image sfimage = run_pipeline(input_uvw, input_model, input_vis, state.range(0), cfg.cell_size, cfg.detection_n_sigma, cfg.analysis_n_sigma, cfg.kernel_support, cfg.kernel_exact, state.range(1));
    }
}

BENCHMARK(pipeline_kernel_oversampling_benchmark)
    ->Args({ 2048, 3 })
    ->Args({ 2048, 5 })
    ->Args({ 2048, 7 })
    ->Args({ 2048, 9 })
    ->Unit(benchmark::kMillisecond);

BENCHMARK(pipeline_kernel_exact_benchmark)
    ->Args({ 2048 })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN()
