/** @file imager_test_benchmark.cpp
 *  @brief Test Imager module performance
 *
 *  @bug No known bugs.
 */

#include <benchmark/benchmark.h>
#include <load_data.h>
#include <load_json_config.h>
#include <stp.h>

std::string data_path(_PIPELINE_DATAPATH);
std::string input_npz("simdata_small.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config.json");
std::string config_file_oversampling("fastimg_oversampling_config.json");

static void imager_test_benchmark(benchmark::State& state)
{
    int image_size = pow(2, state.range(0));

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_model = load_npy_complex_array(data_path + input_npz, "model");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");

    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = input_vis - input_model;

    stp::GaussianSinc kernel_func(cfg.kernel_support);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::image_visibilities(kernel_func, residual_vis, input_uvw, image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling));
    }
}

BENCHMARK(imager_test_benchmark)
    ->Args({ 10 })
    ->Args({ 11 })
    ->Args({ 12 })
    ->Args({ 13 })
    ->Args({ 14 })
    ->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN()
