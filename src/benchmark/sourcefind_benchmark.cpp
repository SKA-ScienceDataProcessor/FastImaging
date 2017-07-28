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
std::string input_npz("simdata_nstep10.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config.json");
std::string config_file_oversampling("fastimg_oversampling_config.json");

static void sourcefind_test_benchmark(benchmark::State& state)
{
    int image_size = pow(2, state.range(0));

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");
    arma::mat input_snr_weights = load_npy_double_array(data_path + input_npz, "snr_weights");
    arma::mat skymodel = load_npy_double_array(data_path + input_npz, "skymodel");

    // Generate model visibilities from the skymodel and UVW-baselines
    arma::cx_mat input_model = stp::generate_visibilities_from_local_skymodel(skymodel, input_uvw);

    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = input_vis - input_model;

    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    stp::GaussianSinc kernel_func(cfg.kernel_support);
    std::pair<arma::Mat<real_t>, arma::Mat<real_t>> result = stp::image_visibilities(kernel_func, residual_vis, input_snr_weights, input_uvw, image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::source_find_image(std::move(result.first), cfg.detection_n_sigma, cfg.analysis_n_sigma,
            cfg.estimate_rms, true, cfg.sigma_clip_iters, cfg.binapprox_median, cfg.compute_barycentre, cfg.generate_labelmap));
        benchmark::ClobberMemory();
    }
}

BENCHMARK(sourcefind_test_benchmark)
    ->DenseRange(10, 14)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN()
