/** @file gridder_benchmark.cpp
 *  @brief Test Gridder module performance
 */

#include <armadillo>
#include <benchmark/benchmark.h>
#include <load_data.h>
#include <load_json_config.h>
#include <stp.h>

std::string data_path(_PIPELINE_DATAPATH);
std::string input_npz("simdata_nstep10.npz");

void load_data(arma::mat& uv_in_pixels, arma::cx_mat& residual_vis, arma::mat& snr_weights, int image_size, double cell_size)
{
    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array<double>(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_vis = load_npy_complex_array<double>(data_path + input_npz, "vis");
    snr_weights = load_npy_double_array<double>(data_path + input_npz, "snr_weights");
    arma::mat skymodel = load_npy_double_array<double>(data_path + input_npz, "skymodel");

    // Generate model visibilities from the skymodel and UVW-baselines
    arma::cx_mat input_model = stp::generate_visibilities_from_local_skymodel(skymodel, input_uvw);

    // Subtract model-generated visibilities from incoming data
    residual_vis = input_vis - input_model;

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cell_size) * double(image_size)));
    uv_in_pixels = (input_uvw / grid_pixel_width_lambda);
    // Remove W column
    uv_in_pixels.shed_col(2);
}

static void gridder_exact_benchmark(benchmark::State& state)
{
    int image_size = pow(2, state.range(0));
    double cell_size = 0.5;
    bool kernel_exact = true;
    int oversampling = 1;
    bool shift_uv = true;
    bool halfplane_gridding = true;

    arma::mat uv_in_pixels;
    arma::cx_mat residual_vis;
    arma::mat snr_weights;
    load_data(uv_in_pixels, residual_vis, snr_weights, image_size, cell_size);

    stp::PSWF kernel_func(state.range(1));

    for (auto _ : state) {
        benchmark::DoNotOptimize(stp::convolve_to_grid<true>(kernel_func, state.range(1), image_size, uv_in_pixels, residual_vis, snr_weights, kernel_exact,
            oversampling, shift_uv, halfplane_gridding));
    }
}

static void gridder_oversampling_benchmark(benchmark::State& state)
{
    int image_size = state.range(0);
    int kernel_support = state.range(1);
    double cell_size = 0.5;
    bool kernel_exact = false;
    int oversampling = 8;
    bool shift_uv = true;
    bool halfplane_gridding = true;

    arma::mat uv_in_pixels;
    arma::cx_mat residual_vis;
    arma::mat snr_weights;
    load_data(uv_in_pixels, residual_vis, snr_weights, image_size, cell_size);

    stp::PSWF kernel_func(kernel_support);

    for (auto _ : state) {
        benchmark::DoNotOptimize(stp::convolve_to_grid<true>(kernel_func, kernel_support, image_size, uv_in_pixels, residual_vis, snr_weights, kernel_exact,
            oversampling, shift_uv, halfplane_gridding));
    }
}

BENCHMARK(gridder_oversampling_benchmark)
    ->RangeMultiplier(2)
    ->Ranges({ { 1 << 10, 1 << 16 }, { 3, 3 } })
    ->Ranges({ { 1 << 10, 1 << 16 }, { 5, 5 } })
    ->Ranges({ { 1 << 10, 1 << 16 }, { 7, 7 } })
    ->Unit(benchmark::kMillisecond);

BENCHMARK(gridder_exact_benchmark)
    ->RangeMultiplier(2)
    ->Ranges({ { 1 << 10, 1 << 16 }, { 3, 3 } })
    ->Ranges({ { 1 << 10, 1 << 16 }, { 5, 5 } })
    ->Ranges({ { 1 << 10, 1 << 16 }, { 7, 7 } })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
