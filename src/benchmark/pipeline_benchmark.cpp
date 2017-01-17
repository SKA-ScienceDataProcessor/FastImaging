/** @file gridder_test_benchmark.cpp
 *  @brief Test Gridder module performance
 *
 *  @bug No known bugs.
 */
#include <armadillo>
#include <benchmark/benchmark.h>
#include <load_data.h>
#include <stp.h>

// Cmake variable
#ifndef _TESTPATH
#define _TESTPATH 0
#endif

std::string path_file(_TESTPATH);
std::string input_npz("simdata.npz");
int image_size = 2048;
double cell_size = 0.1;
double detection_n_sigma = 50.0;
double analysis_n_sigma = 50.0;
int support = 5;

void load_input_data(arma::mat& input_uvw, arma::cx_mat& input_model, arma::cx_mat& input_vis)
{
    cnpy::NpyArray input_uvw1(cnpy::npz_load(path_file + input_npz, "uvw_lambda"));
    cnpy::NpyArray input_model1(cnpy::npz_load(path_file + input_npz, "model"));
    cnpy::NpyArray input_vis1(cnpy::npz_load(path_file + input_npz, "vis"));

    //Load simulated data from input_npz
    input_uvw = load_npy_double_array(input_uvw1);
    input_model = load_npy_complex_array(input_model1);
    input_vis = load_npy_complex_array(input_vis1);
}

stp::source_find_image run_pipeline(arma::mat uvw_lambda, arma::cx_mat model_vis, arma::cx_mat data_vis, int image_size, int cell_size, double detection_n_sigma, double analysis_n_sigma, int support = 3, bool kernel_exact = true, int oversampling = 1)
{
    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = data_vis - model_vis;

    stp::GaussianSinc kernel_func(support);
    std::pair<arma::cx_mat, arma::cx_mat> result = stp::image_visibilities(kernel_func, residual_vis, uvw_lambda, image_size, cell_size, support, kernel_exact, oversampling);

    return stp::source_find_image(arma::real(result.first), detection_n_sigma, analysis_n_sigma, std::experimental::nullopt, true);
}

static void pipeline_kernel_oversampling_benchmark(benchmark::State& state)
{
    int kernel_exact = false;
    arma::mat input_uvw;
    arma::cx_mat input_model;
    arma::cx_mat input_vis;
    load_input_data(input_uvw, input_model, input_vis);

    while (state.KeepRunning())
        stp::source_find_image sfimage = run_pipeline(input_uvw, input_model, input_vis, state.range(0), cell_size, detection_n_sigma, analysis_n_sigma, support, kernel_exact, state.range(1));
}

static void pipeline_kernel_exact_benchmark(benchmark::State& state)
{
    int kernel_exact = true;
    arma::mat input_uvw;
    arma::cx_mat input_model;
    arma::cx_mat input_vis;
    load_input_data(input_uvw, input_model, input_vis);

    while (state.KeepRunning())
        stp::source_find_image sfimage = run_pipeline(input_uvw, input_model, input_vis, state.range(0), cell_size, detection_n_sigma, analysis_n_sigma, support, kernel_exact);
}

BENCHMARK(pipeline_kernel_oversampling_benchmark)
    ->Args({ 2048, 3 })
    ->Args({ 2048, 5 })
    ->Args({ 2048, 7 })
    ->Args({ 2048, 9 });

BENCHMARK(pipeline_kernel_exact_benchmark)
    ->Args({ 2048 });

BENCHMARK_MAIN()
