/** @file fft_c2r_benchmark.cpp
 *  @brief Test fft c2r performance
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
int normalize = true;

static void fft_c2r_test_benchmark(benchmark::State& state, stp::FFTRoutine r_fft)
{
    int image_size = state.range(0);

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array<double>(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_vis = load_npy_complex_array<double>(data_path + input_npz, "vis");
    arma::mat input_snr_weights = load_npy_double_array<double>(data_path + input_npz, "snr_weights");
    arma::mat skymodel = load_npy_double_array<double>(data_path + input_npz, "skymodel");

    // Generate model visibilities from the skymodel and UVW-baselines
    arma::cx_mat input_model = stp::generate_visibilities_from_local_skymodel(skymodel, input_uvw);

    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = input_vis - input_model;

    // Load all configurations from json configuration file
    ConfigurationFile cfg(config_path + config_file_oversampling);

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cfg.img_pars.cell_size) * double(image_size)));
    arma::mat uv_in_pixels = input_uvw / grid_pixel_width_lambda;

    // Remove W column
    uv_in_pixels.shed_col(2);

    stp::PSWF kernel_func(cfg.img_pars.kernel_support);
    stp::GridderOutput gridded_data = stp::convolve_to_grid<false>(kernel_func, cfg.img_pars.kernel_support, image_size, uv_in_pixels, residual_vis,
        input_snr_weights, cfg.img_pars.kernel_exact, cfg.img_pars.oversampling);

    // Compute FFT of first matrix (beam)
    arma::Mat<real_t> fft_result_image;
    // Reuse gridded_data buffer if FFT is INPLACE
    if (r_fft == stp::FFTRoutine::FFTW_WISDOM_INPLACE_FFT) {
        fft_result_image = std::move(arma::Mat<real_t>(reinterpret_cast<real_t*>(gridded_data.vis_grid.memptr()), (gridded_data.vis_grid.n_rows) * 2,
            gridded_data.vis_grid.n_cols, false, false));
    }
    unsigned int fftw_flag = FFTW_ESTIMATE;

    init_fftw(r_fft, cfg.img_pars.fft_wisdom_filename);

    switch (r_fft) {
    case stp::FFTRoutine::FFTW_ESTIMATE_FFT:
        // FFTW implementation using FFTW_ESTIMATE
        for (auto _ : state) {
            benchmark::DoNotOptimize(fft_result_image.memptr());
            stp::fft_fftw_c2r(gridded_data.vis_grid, fft_result_image, r_fft);
            benchmark::ClobberMemory();
        }
        break;
    case stp::FFTRoutine::FFTW_WISDOM_FFT:
        // FFTW implementation using FFTW_WISDOM
        for (auto _ : state) {
            benchmark::DoNotOptimize(fft_result_image.memptr());
            stp::fft_fftw_c2r(gridded_data.vis_grid, fft_result_image, r_fft);
            benchmark::ClobberMemory();
        }
        break;
    case stp::FFTRoutine::FFTW_WISDOM_INPLACE_FFT:
        // FFTW implementation using FFTW_WISDOM_INPLACE
        for (auto _ : state) {
            benchmark::DoNotOptimize(fft_result_image.memptr());
            stp::fft_fftw_c2r(gridded_data.vis_grid, fft_result_image, r_fft);
            benchmark::ClobberMemory();
        }
        break;
    case stp::FFTRoutine::FFTW_MEASURE_FFT:
        fftw_flag = FFTW_MEASURE;
        break;
    case stp::FFTRoutine::FFTW_PATIENT_FFT:
        fftw_flag = FFTW_PATIENT;
        break;
    default:
        break;
    }

    // FFTW implementation using FFTW_MEASURE or FFTW_PATIENT
    if (r_fft == stp::FFTRoutine::FFTW_MEASURE_FFT || r_fft == stp::FFTRoutine::FFTW_PATIENT_FFT) {
        arma::Mat<cx_real_t> input(arma::size(static_cast<arma::Mat<cx_real_t>&>(gridded_data.vis_grid)));
        arma::Mat<real_t> output((input.n_rows % 2 == 0) ? (input.n_rows * 2) : (input.n_rows - 1) * 2, input.n_cols);

#ifdef USE_FLOAT
        fftwf_plan plan = fftwf_plan_dft_c2r_2d(
            input.n_cols, // FFTW uses row-major order, requiring the plan
            input.n_rows, // to be passed the dimensions in reverse.
            reinterpret_cast<fftwf_complex*>(input.memptr()),
            reinterpret_cast<float*>(output.memptr()),
            fftw_flag);

        input = gridded_data.vis_grid;

        for (auto _ : state) {
            benchmark::DoNotOptimize(output.memptr());
            fftwf_execute(plan);
            benchmark::ClobberMemory();
        }
        fftwf_destroy_plan(plan);
#else
        fftw_plan plan = fftw_plan_dft_c2r_2d(
            input.n_cols, // FFTW uses row-major order, requiring the plan
            input.n_rows, // to be passed the dimensions in reverse.
            reinterpret_cast<fftw_complex*>(input.memptr()),
            reinterpret_cast<double*>(output.memptr()),
            fftw_flag);

        input = gridded_data.vis_grid;

        for (auto _ : state) {
            benchmark::DoNotOptimize(output.memptr());
            fftw_execute(plan);
            benchmark::ClobberMemory();
        }
        fftw_destroy_plan(plan);
#endif
    }

#ifdef USE_FLOAT
    fftwf_cleanup_threads();
#else
    fftw_cleanup_threads();
#endif
}

BENCHMARK_CAPTURE(fft_c2r_test_benchmark, FFTW_ESTIMATE_FFT, stp::FFTRoutine::FFTW_ESTIMATE_FFT)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 16)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(fft_c2r_test_benchmark, FFTW_MEASURE_FFT, stp::FFTRoutine::FFTW_MEASURE_FFT)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 16)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(fft_c2r_test_benchmark, FFTW_PATIENT_FFT, stp::FFTRoutine::FFTW_PATIENT_FFT)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 16)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_CAPTURE(fft_c2r_test_benchmark, FFTW_WISDOM_FFT, stp::FFTRoutine::FFTW_WISDOM_FFT)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 16)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
