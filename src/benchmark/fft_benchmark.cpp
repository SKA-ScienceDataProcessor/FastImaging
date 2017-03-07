/** @file imager_test_benchmark.cpp
 *  @brief Test Imager module performance
 *
 *  @bug No known bugs.
 */

#include <../stp/common/fft.h>
#include <benchmark/benchmark.h>
#include <fftw3.h>
#include <load_data.h>
#include <load_json_config.h>
#include <stp.h>
#include <thread>

std::string data_path(_PIPELINE_DATAPATH);
std::string input_npz("simdata_small.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config.json");
std::string config_file_oversampling("fastimg_oversampling_config.json");
int normalize = true;

static void fft_test_benchmark(benchmark::State& state)
{
    stp::fft_function_type f_fft = (stp::fft_function_type)state.range(0);

    int image_size = pow(2, double(state.range(1) + 19) / 2.0);

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_model = load_npy_complex_array(data_path + input_npz, "model");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");

    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = input_vis - input_model;

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cfg.cell_size) * image_size));
    arma::mat uvw_in_pixels = input_uvw / grid_pixel_width_lambda;

    arma::mat uv_in_pixels(uvw_in_pixels.n_rows, 2);
    uv_in_pixels.col(0) = uvw_in_pixels.col(0);
    uv_in_pixels.col(1) = uvw_in_pixels.col(1);

    stp::GaussianSinc kernel_func(cfg.kernel_support);
    std::pair<arma::cx_mat, arma::mat> result = convolve_to_grid(kernel_func, cfg.kernel_support, image_size, uv_in_pixels, residual_vis, cfg.kernel_exact, cfg.oversampling);

    stp::fftshift(result.first, false);
    //stp::fftshift(result.second, false);
    //arma::cx_mat result_beam = arma::conv_to<arma::cx_mat>::from(result.second);

    arma::cx_mat fft_result_image;
    //arma::cx_mat fft_result_beam;

    if (f_fft == stp::FFTW_OPTIMAL) {
        arma::cx_mat output(arma::size(result.first));
        arma::cx_mat input(arma::size(result.first));

        fftw_init_threads();
        fftw_plan_with_nthreads(std::thread::hardware_concurrency());

        int sign = FFTW_BACKWARD;
        fftw_plan plan = fftw_plan_dft_2d(
            input.n_cols, // FFTW uses row-major order, requiring the plan
            input.n_rows, // to be passed the dimensions in reverse.
            reinterpret_cast<fftw_complex*>(input.memptr()),
            reinterpret_cast<fftw_complex*>(output.memptr()),
            sign,
            FFTW_MEASURE);

        input = result.first;

        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(output);

            fftw_execute(plan);
            benchmark::ClobberMemory();
        }
        fftw_destroy_plan(plan);

        fftw_cleanup_threads();

    } else {

        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(fft_result_image);
            //benchmark::DoNotOptimize(fft_result_beam);

            switch (f_fft) {
            case stp::FFTW:
                // FFTW implementation
                fft_result_image = stp::fft_fftw(result.first, true);
                //fft_result_beam = stp::fft_fftw(result_beam, true);
                break;
            case stp::ARMAFFT:
                // Armadillo implementation
                fft_result_image = stp::fft_arma(result.first, true);
                //fft_result_beam = stp::fft_arma(result_beam, true);
                break;
            default:
                assert(0);
            }
            benchmark::ClobberMemory();
        }
    }
}

BENCHMARK(fft_test_benchmark)
    ->Args({ stp::FFTW, 1 })
    ->Args({ stp::FFTW, 2 })
    ->Args({ stp::FFTW, 3 })
    ->Args({ stp::FFTW, 4 })
    ->Args({ stp::FFTW, 5 })
    ->Args({ stp::FFTW, 6 })
    ->Args({ stp::FFTW, 7 })
    ->Args({ stp::FFTW, 8 })
    ->Args({ stp::FFTW, 9 })
    ->Args({ stp::FFTW, 10 })
    ->Args({ stp::ARMAFFT, 1 })
    ->Args({ stp::ARMAFFT, 2 })
    ->Args({ stp::ARMAFFT, 3 })
    ->Args({ stp::ARMAFFT, 4 })
    ->Args({ stp::ARMAFFT, 5 })
    ->Args({ stp::ARMAFFT, 6 })
    ->Args({ stp::ARMAFFT, 7 })
    ->Args({ stp::ARMAFFT, 8 })
    ->Args({ stp::ARMAFFT, 9 })
    ->Args({ stp::ARMAFFT, 10 })
    ->Args({ stp::FFTW_OPTIMAL, 1 })
    ->Args({ stp::FFTW_OPTIMAL, 2 })
    ->Args({ stp::FFTW_OPTIMAL, 3 })
    ->Args({ stp::FFTW_OPTIMAL, 4 })
    ->Args({ stp::FFTW_OPTIMAL, 5 })
    ->Args({ stp::FFTW_OPTIMAL, 6 })
    ->Args({ stp::FFTW_OPTIMAL, 7 })
    ->Args({ stp::FFTW_OPTIMAL, 8 })
    ->Args({ stp::FFTW_OPTIMAL, 9 })
    ->Args({ stp::FFTW_OPTIMAL, 10 })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN()
