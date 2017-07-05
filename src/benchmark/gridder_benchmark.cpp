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
std::string input_npz("simdata_nstep10.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config.json");
std::string config_file_oversampling("fastimg_oversampling_config.json");

void load_data(arma::mat& uv_in_pixels, arma::cx_mat& residual_vis, int image_size, double cell_size, int kernel_support)
{
    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");
    arma::mat skymodel = load_npy_double_array(data_path + input_npz, "skymodel");

    // Generate model visibilities from the skymodel and UVW-baselines
    arma::cx_mat input_model = stp::generate_visibilities_from_local_skymodel(skymodel, input_uvw);

    // Subtract model-generated visibilities from incoming data
    residual_vis = input_vis - input_model;

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cell_size) * double(image_size)));
    uv_in_pixels = (input_uvw / grid_pixel_width_lambda);

    // Remove W column
    uv_in_pixels.shed_col(2);

    // If a visibility point is located in the top half-plane, move it to the bottom half-plane to a symmetric position with respect to the matrix centre (0,0)
    stp::convert_to_halfplane_visibilities(uv_in_pixels, residual_vis, kernel_support);
}

static void gridder_exact_benchmark(benchmark::State& state)
{
    int image_size = pow(2, state.range(0));

    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_exact);

    arma::mat uv_in_pixels;
    arma::cx_mat residual_vis;
    load_data(uv_in_pixels, residual_vis, image_size, cfg.cell_size, cfg.kernel_support);

    stp::GaussianSinc kernel_func(state.range(1));

    while (state.KeepRunning())
        benchmark::DoNotOptimize(stp::convolve_to_grid(kernel_func, state.range(1), image_size, uv_in_pixels, residual_vis, cfg.kernel_exact, 1));
}

static void gridder_oversampling_benchmark(benchmark::State& state)
{
    int image_size = pow(2, state.range(0));

    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    arma::mat uv_in_pixels;
    arma::cx_mat residual_vis;
    load_data(uv_in_pixels, residual_vis, image_size, cfg.cell_size, cfg.kernel_support);

    stp::GaussianSinc kernel_func(state.range(1));

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::convolve_to_grid(kernel_func, state.range(1), image_size, uv_in_pixels, residual_vis, cfg.kernel_exact, cfg.oversampling));
    }
}

BENCHMARK(gridder_oversampling_benchmark)
    ->Args({ 10, 3 })
    ->Args({ 11, 3 })
    ->Args({ 12, 3 })
    ->Args({ 13, 3 })
    ->Args({ 14, 3 })
    ->Args({ 15, 3 })
    ->Args({ 10, 5 })
    ->Args({ 11, 5 })
    ->Args({ 12, 5 })
    ->Args({ 13, 5 })
    ->Args({ 14, 5 })
    ->Args({ 15, 5 })
    ->Args({ 10, 7 })
    ->Args({ 11, 7 })
    ->Args({ 12, 7 })
    ->Args({ 13, 7 })
    ->Args({ 14, 7 })
    ->Args({ 15, 7 })
    ->Unit(benchmark::kMillisecond);

BENCHMARK(gridder_exact_benchmark)
    ->Args({ 10, 3 })
    ->Args({ 11, 3 })
    ->Args({ 12, 3 })
    ->Args({ 13, 3 })
    ->Args({ 14, 3 })
    ->Args({ 15, 3 })
    ->Args({ 10, 5 })
    ->Args({ 11, 5 })
    ->Args({ 12, 5 })
    ->Args({ 13, 5 })
    ->Args({ 14, 5 })
    ->Args({ 15, 5 })
    ->Args({ 10, 7 })
    ->Args({ 11, 7 })
    ->Args({ 12, 7 })
    ->Args({ 13, 7 })
    ->Args({ 14, 7 })
    ->Args({ 15, 7 })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN()
