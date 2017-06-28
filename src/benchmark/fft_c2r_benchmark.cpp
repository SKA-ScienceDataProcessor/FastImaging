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
std::string input_npz("simdata_nstep10.npz");
std::string config_path(_PIPELINE_CONFIGPATH);
std::string config_file_exact("fastimg_exact_config.json");
std::string config_file_oversampling("fastimg_oversampling_config.json");
std::string wisdom_path(_WISDOM_FILEPATH);
int normalize = true;

static void fft_c2r_test_benchmark(benchmark::State& state)
{
    stp::FFTRoutine r_fft = (stp::FFTRoutine)state.range(0);

    int image_size = pow(2, double(state.range(1) + 19) / 2.0);

    std::string wisdom_filename;
    if (r_fft == stp::FFTW_WISDOM_INPLACE_FFT) {
        wisdom_filename = wisdom_path + "WisdomFile_rib" + std::to_string(image_size) + "x" + std::to_string(image_size) + ".fftw";
    } else {
        wisdom_filename = wisdom_path + "WisdomFile_rob" + std::to_string(image_size) + "x" + std::to_string(image_size) + ".fftw";
    }

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_model = load_npy_complex_array(data_path + input_npz, "model");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");

    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = input_vis - input_model;

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cfg.cell_size) * double(image_size)));
    arma::mat uv_in_pixels = input_uvw / grid_pixel_width_lambda;

    // Remove W column
    uv_in_pixels.shed_col(2);

    // If a visibility point is located in the left half-plane, move it to the right half-plane by inverting both x and y values
    for (size_t i = 0; i < uv_in_pixels.n_rows; ++i) {
        if (uv_in_pixels.at(i, 1) < 0.0) {
            uv_in_pixels.at(i) *= (-1);
            residual_vis.at(i) = std::conj(residual_vis.at(i));
        }
    }

    stp::GaussianSinc kernel_func(cfg.kernel_support);
    auto gridded_data = convolve_to_grid(kernel_func, cfg.kernel_support, image_size, uv_in_pixels, residual_vis, cfg.kernel_exact, cfg.oversampling);

    // Compute FFT of first matrix (beam)
    if (cfg.kernel_exact) {
        // Shift beam if kernel_exact is true
        stp::fftshift(gridded_data.first, false);
    }
    arma::Mat<real_t> fft_result_image;
    // Reuse gridded_data buffer if FFT is INPLACE
    if (r_fft == stp::FFTW_WISDOM_INPLACE_FFT) {
        fft_result_image = std::move(arma::Mat<real_t>(reinterpret_cast<real_t*>(gridded_data.first.memptr()), (gridded_data.first.n_rows) * 2, gridded_data.first.n_cols, false, false));
    }
    unsigned int fftw_flag = FFTW_ESTIMATE;

    switch (r_fft) {
    case stp::FFTW_ESTIMATE_FFT:
        // FFTW implementation using FFTW_ESTIMATE
        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(fft_result_image);
            stp::fft_fftw_c2r(gridded_data.first, fft_result_image, r_fft);
            benchmark::ClobberMemory();
        }
        break;
    case stp::FFTW_WISDOM_FFT:
        // FFTW implementation using FFTW_WISDOM
        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(fft_result_image);
            stp::fft_fftw_c2r(gridded_data.first, fft_result_image, r_fft, wisdom_filename);
            benchmark::ClobberMemory();
        }
        break;
    case stp::FFTW_WISDOM_INPLACE_FFT:
        // FFTW implementation using FFTW_WISDOM_INPLACE
        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(fft_result_image);
            stp::fft_fftw_c2r(gridded_data.first, fft_result_image, r_fft, wisdom_filename);
            benchmark::ClobberMemory();
        }
        break;
    case stp::FFTW_MEASURE_FFT:
        fftw_flag = FFTW_MEASURE;
        break;
    case stp::FFTW_PATIENT_FFT:
        fftw_flag = FFTW_PATIENT;
        break;
    default:
        break;
    }

    // FFTW implementation using FFTW_MEASURE or FFTW_PATIENT
    if (r_fft == stp::FFTW_MEASURE_FFT || r_fft == stp::FFTW_PATIENT_FFT) {
        arma::Mat<cx_real_t> input(arma::size(static_cast<arma::Mat<cx_real_t>&>(gridded_data.first)));
        arma::Mat<real_t> output((input.n_rows % 2 == 0) ? (input.n_rows * 2) : (input.n_rows - 1) * 2, input.n_cols);

#ifdef USE_FLOAT
        fftwf_init_threads();
        fftwf_plan_with_nthreads(std::thread::hardware_concurrency());

        fftwf_plan plan = fftwf_plan_dft_c2r_2d(
            input.n_cols, // FFTW uses row-major order, requiring the plan
            input.n_rows, // to be passed the dimensions in reverse.
            reinterpret_cast<fftwf_complex*>(input.memptr()),
            reinterpret_cast<float*>(output.memptr()),
            fftw_flag);

        input = gridded_data.first;

        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(output);
            fftwf_execute(plan);
            benchmark::ClobberMemory();
        }
        fftwf_destroy_plan(plan);
        fftwf_cleanup_threads();
#else
        fftw_init_threads();
        fftw_plan_with_nthreads(std::thread::hardware_concurrency());

        fftw_plan plan = fftw_plan_dft_c2r_2d(
            input.n_cols, // FFTW uses row-major order, requiring the plan
            input.n_rows, // to be passed the dimensions in reverse.
            reinterpret_cast<fftw_complex*>(input.memptr()),
            reinterpret_cast<double*>(output.memptr()),
            fftw_flag);

        input = gridded_data.first;

        while (state.KeepRunning()) {
            benchmark::DoNotOptimize(output);
            fftw_execute(plan);
            benchmark::ClobberMemory();
        }
        fftw_destroy_plan(plan);
        fftw_cleanup_threads();
#endif
    }
}

BENCHMARK(fft_c2r_test_benchmark)
    ->Args({ stp::FFTW_ESTIMATE_FFT, 1 })
    ->Args({ stp::FFTW_ESTIMATE_FFT, 2 })
    ->Args({ stp::FFTW_ESTIMATE_FFT, 3 })
    ->Args({ stp::FFTW_ESTIMATE_FFT, 4 })
    ->Args({ stp::FFTW_ESTIMATE_FFT, 5 })
    ->Args({ stp::FFTW_ESTIMATE_FFT, 6 })
    ->Args({ stp::FFTW_ESTIMATE_FFT, 7 })
    ->Args({ stp::FFTW_ESTIMATE_FFT, 9 })
    ->Args({ stp::FFTW_ESTIMATE_FFT, 11 })
    ->Args({ stp::FFTW_ESTIMATE_FFT, 13 })
    ->Args({ stp::FFTW_MEASURE_FFT, 1 })
    ->Args({ stp::FFTW_MEASURE_FFT, 2 })
    ->Args({ stp::FFTW_MEASURE_FFT, 3 })
    ->Args({ stp::FFTW_MEASURE_FFT, 4 })
    ->Args({ stp::FFTW_MEASURE_FFT, 5 })
    ->Args({ stp::FFTW_MEASURE_FFT, 6 })
    ->Args({ stp::FFTW_MEASURE_FFT, 7 })
    ->Args({ stp::FFTW_MEASURE_FFT, 9 })
    ->Args({ stp::FFTW_MEASURE_FFT, 11 })
    ->Args({ stp::FFTW_MEASURE_FFT, 13 })
    ->Args({ stp::FFTW_PATIENT_FFT, 1 })
    ->Args({ stp::FFTW_PATIENT_FFT, 2 })
    ->Args({ stp::FFTW_PATIENT_FFT, 3 })
    ->Args({ stp::FFTW_PATIENT_FFT, 4 })
    ->Args({ stp::FFTW_PATIENT_FFT, 5 })
    ->Args({ stp::FFTW_PATIENT_FFT, 6 })
    ->Args({ stp::FFTW_PATIENT_FFT, 7 })
    ->Args({ stp::FFTW_PATIENT_FFT, 9 })
    ->Args({ stp::FFTW_PATIENT_FFT, 11 })
    ->Args({ stp::FFTW_PATIENT_FFT, 13 })
    ->Args({ stp::FFTW_WISDOM_FFT, 1 })
    ->Args({ stp::FFTW_WISDOM_FFT, 2 })
    ->Args({ stp::FFTW_WISDOM_FFT, 3 })
    ->Args({ stp::FFTW_WISDOM_FFT, 4 })
    ->Args({ stp::FFTW_WISDOM_FFT, 5 })
    ->Args({ stp::FFTW_WISDOM_FFT, 6 })
    ->Args({ stp::FFTW_WISDOM_FFT, 7 })
    ->Args({ stp::FFTW_WISDOM_FFT, 9 })
    ->Args({ stp::FFTW_WISDOM_FFT, 11 })
    ->Args({ stp::FFTW_WISDOM_FFT, 13 })
    ->Args({ stp::FFTW_WISDOM_INPLACE_FFT, 1 })
    ->Args({ stp::FFTW_WISDOM_INPLACE_FFT, 2 })
    ->Args({ stp::FFTW_WISDOM_INPLACE_FFT, 3 })
    ->Args({ stp::FFTW_WISDOM_INPLACE_FFT, 4 })
    ->Args({ stp::FFTW_WISDOM_INPLACE_FFT, 5 })
    ->Args({ stp::FFTW_WISDOM_INPLACE_FFT, 6 })
    ->Args({ stp::FFTW_WISDOM_INPLACE_FFT, 7 })
    ->Args({ stp::FFTW_WISDOM_INPLACE_FFT, 9 })
    ->Args({ stp::FFTW_WISDOM_INPLACE_FFT, 11 })
    ->Args({ stp::FFTW_WISDOM_INPLACE_FFT, 13 })
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN()
