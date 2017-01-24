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
    arma::mat input_uvw;
    arma::cx_mat input_model, input_vis;
    load_npz_simdata(data_path + input_npz, input_uvw, input_model, input_vis);

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
    std::pair<arma::cx_mat, arma::cx_mat> result = convolve_to_grid(kernel_func, cfg.kernel_support, cfg.image_size, uv_in_pixels, residual_vis, cfg.kernel_exact, cfg.oversampling);

    while (state.KeepRunning()) {
        arma::cx_mat shifted_image = stp::fftshift(result.first, false);
        arma::cx_mat shifted_beam = stp::fftshift(result.second, false);

        arma::cx_mat fft_result_image;
        arma::cx_mat fft_result_beam;

        benchmark::DoNotOptimize(fft_result_image);
        benchmark::DoNotOptimize(fft_result_beam);

        switch (f_fft) {
        case stp::FFTW:
            // FFTW implementation
            fft_result_image = stp::fftshift(stp::fft_fftw(shifted_image, true));
            fft_result_beam = stp::fftshift(stp::fft_fftw(shifted_beam, true));
            break;
        case stp::ARMAFFT:
            // Armadillo implementation
            fft_result_image = stp::fftshift(stp::fft_arma(shifted_image, true));
            fft_result_beam = stp::fftshift(stp::fft_arma(shifted_beam, true));
            break;
        default:
            assert(0);
        }

        // Normalize image and beam such that the beam peaks at a value of 1.0 Jansky.
        if (normalize == true) {
            double beam_max = arma::real(fft_result_beam).max();
            fft_result_beam /= beam_max;
            fft_result_image /= beam_max;
        }
        benchmark::ClobberMemory();
    }
}

BENCHMARK(fft_test_benchmark)
    ->Args({ stp::FFTW })
    ->Args({ stp::ARMAFFT })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN()
