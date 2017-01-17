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

void prepare_gridder(arma::mat& uv_in_pixels, arma::cx_mat& residual_vis)
{
    arma::mat input_uvw;
    arma::cx_mat input_model;
    arma::cx_mat input_vis;
    load_input_data(input_uvw, input_model, input_vis);

    // Subtract model-generated visibilities from incoming data
    residual_vis = input_vis - input_model;

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cell_size) * image_size));
    arma::mat uvw_in_pixels = input_uvw / grid_pixel_width_lambda;

    uv_in_pixels.resize(uvw_in_pixels.n_rows, 2);
    uv_in_pixels.col(0) = uvw_in_pixels.col(0);
    uv_in_pixels.col(1) = uvw_in_pixels.col(1);
}

static void gridder_kernel_oversampling_benchmark(benchmark::State& state)
{
    int kernel_exact = false;

    stp::GaussianSinc kernel_func(support);
    arma::mat uv_in_pixels;
    arma::cx_mat residual_vis;

    prepare_gridder(uv_in_pixels, residual_vis);

    while (state.KeepRunning())
        std::pair<arma::cx_mat, arma::cx_mat> result = convolve_to_grid(kernel_func, state.range(1), image_size, uv_in_pixels, residual_vis, kernel_exact, state.range(0));
}

static void gridder_kernel_exact_benchmark(benchmark::State& state)
{
    int kernel_exact = true;

    stp::GaussianSinc kernel_func(support);
    arma::mat uv_in_pixels;
    arma::cx_mat residual_vis;

    prepare_gridder(uv_in_pixels, residual_vis);

    while (state.KeepRunning())
        std::pair<arma::cx_mat, arma::cx_mat> result = convolve_to_grid(kernel_func, state.range(0), image_size, uv_in_pixels, residual_vis, kernel_exact, 1);
}

BENCHMARK(gridder_kernel_oversampling_benchmark)
    ->Args({ 3, 3 })
    ->Args({ 5, 3 })
    ->Args({ 7, 3 })
    ->Args({ 9, 3 })
    ->Args({ 3, 7 })
    ->Args({ 5, 7 })
    ->Args({ 7, 7 })
    ->Args({ 9, 7 });

BENCHMARK(gridder_kernel_exact_benchmark)
    ->Args({ 3 })
    ->Args({ 5 })
    ->Args({ 7 });

BENCHMARK_MAIN()
