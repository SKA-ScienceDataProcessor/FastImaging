/** @file sourcefind_test_benchmark.cpp
 *  @brief Test SourceFindImage module performance
 *
 *  @bug No known bugs.
 */

#include <benchmark/benchmark.h>
#include <load_data.h>
#include <load_json_config.h>
#include <stp.h>

std::string data_path(_PIPELINE_DATAPATH);
std::string input_npz("simdata.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config2.json");
std::string config_file_oversampling("fastimg_oversampling_config2.json");

static void sourcefind_test_benchmark(benchmark::State& state)
{
    //Load simulated data from input_npz
    arma::mat input_uvw;
    arma::cx_mat input_model, input_vis;
    load_npz_simdata(data_path + input_npz, input_uvw, input_model, input_vis);

    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = input_vis - input_model;

    stp::GaussianSinc kernel_func(cfg.kernel_support);
    std::pair<arma::cx_mat, arma::cx_mat> result = stp::image_visibilities(kernel_func, residual_vis, input_uvw, cfg.image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::source_find_image(arma::real(result.first), cfg.detection_n_sigma, cfg.analysis_n_sigma, 0.0, true));
    }
}

BENCHMARK(sourcefind_test_benchmark)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN()
