/** @file pipeline_benchmark.cpp
 *  @brief Test pipeline performance
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
std::string input_npz("simdata_nstep10.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config.json");
std::string config_file_oversampling("fastimg_oversampling_config.json");
std::string wisdom_path(_WISDOM_FILEPATH);

stp::SourceFindImage run_pipeline(arma::mat& uvw_lambda, arma::cx_mat& residual_vis, arma::mat& vis_weights, int image_size, ConfigurationFile& cfg)
{
    std::string wisdom_filename = wisdom_path + "WisdomFile_rob" + std::to_string(image_size) + "x" + std::to_string(image_size) + ".fftw";

    stp::GaussianSinc kernel_func(cfg.kernel_support);
    std::pair<arma::Mat<real_t>, arma::Mat<real_t>> result = stp::image_visibilities(kernel_func, residual_vis, vis_weights, uvw_lambda, image_size,
        cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling, cfg.generate_beam, stp::FFTRoutine::FFTW_WISDOM_FFT, wisdom_filename);
    result.second.reset();

    return stp::SourceFindImage(std::move(result.first), cfg.detection_n_sigma, cfg.analysis_n_sigma, cfg.estimate_rms,
        true, cfg.sigma_clip_iters, cfg.binapprox_median, cfg.gaussian_fitting, cfg.generate_labelmap,
        cfg.ceres_diffmethod, cfg.ceres_solvertype);
}

static void pipeline_kernel_exact_benchmark(benchmark::State& state)
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
    ConfigurationFile cfg(config_path + config_file_exact);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(run_pipeline(input_uvw, residual_vis, input_snr_weights, image_size, cfg));
    }
}

static void pipeline_kernel_oversampling_benchmark(benchmark::State& state)
{
    int image_size = pow(2, state.range(0));

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");
    arma::mat input_snr_weights = load_npy_double_array(data_path + input_npz, "snr_weights");
    arma::mat skymodel = load_npy_double_array(data_path + input_npz, "skymodel");

    // Generate model visibilities from the skymodel and UVW-baselines
    arma::cx_mat input_model = stp::generate_visibilities_from_local_skymodel(skymodel, input_uvw);

    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = input_vis - input_model;

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(run_pipeline(input_uvw, residual_vis, input_snr_weights, image_size, cfg));
    }
}

BENCHMARK(pipeline_kernel_oversampling_benchmark)
    ->DenseRange(10, 14) // 10,11,12,13,14
    ->Unit(benchmark::kMillisecond);

BENCHMARK(pipeline_kernel_exact_benchmark)
    ->DenseRange(10, 14)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN()
