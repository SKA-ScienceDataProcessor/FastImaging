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
std::string input_npz("simdata_nstep10.npz");
std::string wisdom_path(_WISDOM_FILEPATH);

stp::source_find_image run_pipeline(arma::mat uvw_lambda, arma::cx_mat residual_vis, int image_size, double cell_size, double detection_n_sigma, double analysis_n_sigma, int support = 3, bool kernel_exact = true, int oversampling = 1)
{
    std::string wisdom_filename = wisdom_path + "WisdomFile_cof" + std::to_string(image_size) + "x" + std::to_string(image_size) + ".fftw";
    std::string wisdom_filename_r2c = wisdom_path + "WisdomFile_rof" + std::to_string(image_size) + "x" + std::to_string(image_size) + ".fftw";

    stp::GaussianSinc kernel_func(support);
    std::pair<arma::Mat<cx_real_t>, arma::Mat<cx_real_t> > result = stp::image_visibilities(kernel_func, std::move(residual_vis), std::move(uvw_lambda),
        image_size, cell_size, support, kernel_exact, oversampling, true, stp::FFTW_WISDOM_FFT, wisdom_filename, wisdom_filename_r2c);

    return stp::source_find_image(std::move(arma::real(result.first)), detection_n_sigma, analysis_n_sigma);
}

static void pipeline_kernel_exact_benchmark(benchmark::State& state)
{
    int image_size = pow(2, state.range(0));

    // Configuration parameters
    double cell_size = 0.5;
    double detection_n_sigma = 50.0;
    double analysis_n_sigma = 50.0;
    int kernel_support = 3;
    bool kernel_exact = true;
    int oversampling = 1;

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_model = load_npy_complex_array(data_path + input_npz, "model");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");
    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = input_vis - input_model;

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(run_pipeline(input_uvw, residual_vis, image_size, cell_size, detection_n_sigma, analysis_n_sigma, kernel_support, kernel_exact, oversampling));
    }
}

static void pipeline_kernel_oversampling_benchmark(benchmark::State& state)
{
    int image_size = pow(2, state.range(0));

    // Configuration parameters
    double cell_size = 0.5;
    double detection_n_sigma = 50.0;
    double analysis_n_sigma = 50.0;
    int kernel_support = 3;
    bool kernel_exact = false;
    int oversampling = 9;

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_model = load_npy_complex_array(data_path + input_npz, "model");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");
    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = input_vis - input_model;

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(run_pipeline(input_uvw, residual_vis, image_size, cell_size, detection_n_sigma, analysis_n_sigma, kernel_support, kernel_exact, oversampling));
    }
}

BENCHMARK(pipeline_kernel_oversampling_benchmark)
    ->DenseRange(10, 14) // 10,11,12,13,14
    ->Unit(benchmark::kMillisecond);

BENCHMARK(pipeline_kernel_exact_benchmark)
    ->DenseRange(10, 14)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN()
