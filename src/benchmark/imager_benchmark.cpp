/** @file imager_benchmark.cpp
 *  @brief Test Imager module performance
 */

#include <benchmark/benchmark.h>
#include <load_data.h>
#include <load_json_config.h>
#include <stp.h>

std::string data_path(_PIPELINE_DATAPATH);
std::string input_npz("simdata_nstep10.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config.json");
std::string config_file_oversampling("fastimg_oversampling_config.json");

static void imager_test_benchmark(benchmark::State& state)
{
    int image_size = state.range(0);

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array<double>(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_vis = load_npy_complex_array<double>(data_path + input_npz, "vis");
    arma::mat input_snr_weights = load_npy_double_array<double>(data_path + input_npz, "snr_weights");
    arma::mat skymodel = load_npy_double_array<double>(data_path + input_npz, "skymodel");

    // Generate model visibilities from the skymodel and UVW-baselines
    arma::cx_mat input_model = stp::generate_visibilities_from_local_skymodel(skymodel, input_uvw);

    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = input_vis - input_model;

    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);
    cfg.img_pars.image_size = image_size;

    stp::PSWF kernel_func(cfg.img_pars.kernel_support);

    for (auto _ : state) {
        benchmark::DoNotOptimize(stp::image_visibilities(kernel_func, residual_vis, input_snr_weights, input_uvw, cfg.img_pars));
    }
}

BENCHMARK(imager_test_benchmark)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 16)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
