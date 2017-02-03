/** @file imager_test_benchmark.cpp
 *  @brief Test Imager module performance
 *
 *  @bug No known bugs.
 */

#include <../stp/common/fft.h>
#include <benchmark/benchmark.h>
#include <load_data.h>
#include <load_json_config.h>
#include <stp.h>

std::string data_path(_PIPELINE_DATAPATH);
std::string input_npz("simdata.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config2.json");
std::string config_file_oversampling("fastimg_oversampling_config2.json");
int normalize = true;

static void fft_test_benchmark(benchmark::State& state)
{
    stp::fft_function_type f_fft = (stp::fft_function_type)state.range(0);

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_model = load_npy_complex_array(data_path + input_npz, "model");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");

    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = input_vis - input_model;

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cfg.cell_size) * cfg.image_size));
    arma::mat uvw_in_pixels = input_uvw / grid_pixel_width_lambda;

    arma::mat uv_in_pixels(uvw_in_pixels.n_rows, 2);
    uv_in_pixels.col(0) = uvw_in_pixels.col(0);
    uv_in_pixels.col(1) = uvw_in_pixels.col(1);

    stp::GaussianSinc kernel_func(cfg.kernel_support);
    std::pair<arma::cx_mat, arma::mat> result = convolve_to_grid(kernel_func, cfg.kernel_support, cfg.image_size, uv_in_pixels, residual_vis, cfg.kernel_exact, cfg.oversampling);

    stp::fftshift(result.first, false);
    stp::fftshift(result.second, false);

    arma::cx_mat fft_result_image;
    arma::cx_mat fft_result_beam;
    fft_result_beam = arma::conv_to<arma::cx_mat>::from(result.second);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(fft_result_image);
        benchmark::DoNotOptimize(fft_result_beam);

        switch (f_fft) {
        case stp::FFTW:
            // FFTW implementation
            fft_result_image = stp::fft_fftw(result.first, true);
            fft_result_beam = stp::fft_fftw(fft_result_beam, true);
            break;
        case stp::ARMAFFT:
            // Armadillo implementation
            fft_result_image = stp::fft_arma(result.first, true);
            fft_result_beam = stp::fft_arma(fft_result_beam, true);
            break;
        default:
            assert(0);
        }
        benchmark::ClobberMemory();
    }
}

BENCHMARK(fft_test_benchmark)
    ->Args({ stp::FFTW })
    ->Args({ stp::ARMAFFT })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN()
