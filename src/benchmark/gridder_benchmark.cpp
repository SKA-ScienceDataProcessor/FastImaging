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

std::string data_path(_PIPELINE_DATAPATH);
std::string input_npz("simdata.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config2.json");
std::string config_file_oversampling("fastimg_oversampling_config2.json");

void prepare_gridder(arma::mat& uv_in_pixels, arma::cx_mat& residual_vis, int image_size, double cell_size)
{
    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_model = load_npy_complex_array(data_path + input_npz, "model");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");

    // Subtract model-generated visibilities from incoming data
    residual_vis = input_vis - input_model;

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cell_size) * image_size));
    arma::mat uvw_in_pixels = input_uvw / grid_pixel_width_lambda;

    uv_in_pixels.resize(uvw_in_pixels.n_rows, 2);
    uv_in_pixels.col(0) = uvw_in_pixels.col(0);
    uv_in_pixels.col(1) = uvw_in_pixels.col(1);
}

static void gridder_kernel_exact_benchmark(benchmark::State& state)
{
    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_exact);

    arma::mat uv_in_pixels;
    arma::cx_mat residual_vis;
    prepare_gridder(uv_in_pixels, residual_vis, cfg.image_size, cfg.cell_size);

    stp::GaussianSinc kernel_func(cfg.kernel_support);
    std::pair<arma::cx_mat, arma::mat> result;

    while (state.KeepRunning())
        benchmark::DoNotOptimize(result = convolve_to_grid(kernel_func, state.range(0), cfg.image_size, uv_in_pixels, residual_vis, cfg.kernel_exact, 1));
}

static void gridder_kernel_oversampling_benchmark(benchmark::State& state)
{
    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    arma::mat uv_in_pixels;
    arma::cx_mat residual_vis;
    prepare_gridder(uv_in_pixels, residual_vis, cfg.image_size, cfg.cell_size);

    stp::GaussianSinc kernel_func(cfg.kernel_support);
    std::pair<arma::cx_mat, arma::mat> result;

    while (state.KeepRunning())
        benchmark::DoNotOptimize(result = convolve_to_grid(kernel_func, state.range(1), cfg.image_size, uv_in_pixels, residual_vis, cfg.kernel_exact, state.range(0)));
}

BENCHMARK(gridder_kernel_oversampling_benchmark)
    ->Args({ 3, 3 })
    ->Args({ 5, 3 })
    ->Args({ 7, 3 })
    ->Args({ 9, 3 })
    ->Args({ 3, 5 })
    ->Args({ 5, 5 })
    ->Args({ 7, 5 })
    ->Args({ 9, 5 })
    ->Args({ 3, 7 })
    ->Args({ 5, 7 })
    ->Args({ 7, 7 })
    ->Args({ 9, 7 })
    ->Unit(benchmark::kMillisecond);

BENCHMARK(gridder_kernel_exact_benchmark)
    ->Args({ 3 })
    ->Args({ 5 })
    ->Args({ 7 })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN()
