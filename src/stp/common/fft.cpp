#include "fft.h"
#include <cassert>
#include <fftw3.h>
#include <thread>

namespace stp {

void fft_fftw_c2r(arma::Mat<cx_real_t>& input, arma::Mat<real_t>& output, FFTRoutine r_fft, const std::string& wisdom_filename)
{
    size_t n_rows = (input.n_rows % 2 == 0) ? (input.n_rows * 2) : (input.n_rows - 1) * 2;
    size_t n_cols = input.n_cols;

    if (reinterpret_cast<real_t*>(input.memptr()) != output.memptr()) {
        output.set_size(n_rows, n_cols);
    }
    unsigned int fftw_flag = FFTW_ESTIMATE;

#ifdef USE_FLOAT
    fftwf_init_threads();
    fftwf_plan_with_nthreads(std::thread::hardware_concurrency());
#else
    fftw_init_threads();
    fftw_plan_with_nthreads(std::thread::hardware_concurrency());
#endif

    switch (r_fft) {
    case FFTW_ESTIMATE_FFT:
        fftw_flag = FFTW_ESTIMATE;
        break;
    case FFTW_MEASURE_FFT:
        // Do not use this mode as the plan generation is very slow
        fftw_flag = FFTW_MEASURE;
        assert(0);
        break;
    case FFTW_PATIENT_FFT:
        // Do not use this mode as the plan generation is extremely slow
        fftw_flag = FFTW_PATIENT;
        assert(0);
        break;
    case FFTW_WISDOM_FFT:
    case FFTW_WISDOM_INPLACE_FFT:
        fftw_flag = FFTW_WISDOM_ONLY;
#ifdef USE_FLOAT
        if (!fftwf_import_wisdom_from_filename(wisdom_filename.c_str())) {
#else
        if (!fftw_import_wisdom_from_filename(wisdom_filename.c_str())) {
#endif
        }
        break;
    default:
        assert(0);
        break;
    }

#ifdef USE_FLOAT

    fftwf_plan plan = fftwf_plan_dft_c2r_2d(
        n_cols, // FFTW uses row-major order, requiring the plan
        n_rows, // to be passed the dimensions in reverse.
        reinterpret_cast<fftwf_complex*>(input.memptr()),
        reinterpret_cast<float*>(output.memptr()),
        fftw_flag);

    assert(plan != NULL);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    fftwf_cleanup_threads();
#else
    fftw_plan plan = fftw_plan_dft_c2r_2d(
        n_cols, // FFTW uses row-major order, requiring the plan
        n_rows, // to be passed the dimensions in reverse.
        reinterpret_cast<fftw_complex*>(input.memptr()),
        reinterpret_cast<double*>(output.memptr()),
        fftw_flag);

    assert(plan != NULL);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    fftw_cleanup_threads();
#endif
}

void fft_fftw_r2c(arma::Mat<real_t>& input, arma::Mat<cx_real_t>& output, FFTRoutine r_fft, const std::string& wisdom_filename)
{
    size_t n_rows = input.n_rows / 2 + 1;
    size_t n_cols = input.n_cols;
    if (input.memptr() != reinterpret_cast<real_t*>(output.memptr())) {
        output.set_size(n_rows, n_cols);
    }
    unsigned int fftw_flag = FFTW_ESTIMATE;

#ifdef USE_FLOAT
    fftwf_init_threads();
    fftwf_plan_with_nthreads(std::thread::hardware_concurrency());
#else
    fftw_init_threads();
    fftw_plan_with_nthreads(std::thread::hardware_concurrency());
#endif

    switch (r_fft) {
    case FFTW_ESTIMATE_FFT:
        fftw_flag = FFTW_ESTIMATE;
        break;
    case FFTW_MEASURE_FFT:
        // Do not use this mode as the plan generation is very slow
        fftw_flag = FFTW_MEASURE;
        assert(0);
        break;
    case FFTW_PATIENT_FFT:
        // Do not use this mode as the plan generation is extremely slow
        fftw_flag = FFTW_PATIENT;
        assert(0);
        break;
    case FFTW_WISDOM_FFT:
    case FFTW_WISDOM_INPLACE_FFT:
        fftw_flag = FFTW_WISDOM_ONLY;
#ifdef USE_FLOAT
        if (!fftwf_import_wisdom_from_filename(wisdom_filename.c_str())) {
#else
        if (!fftw_import_wisdom_from_filename(wisdom_filename.c_str())) {
#endif
            assert(0);
        }
        break;
    default:
        assert(0);
        break;
    }

#ifdef USE_FLOAT
    fftwf_plan plan = fftwf_plan_dft_r2c_2d(
        n_cols, // FFTW uses row-major order, requiring the plan
        n_rows, // to be passed the dimensions in reverse.
        reinterpret_cast<float*>(input.memptr()),
        reinterpret_cast<fftwf_complex*>(output.memptr()),
        fftw_flag);

    assert(plan != NULL);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    fftwf_cleanup_threads();
#else
    fftw_plan plan = fftw_plan_dft_r2c_2d(
        n_cols, // FFTW uses row-major order, requiring the plan
        n_rows, // to be passed the dimensions in reverse.
        reinterpret_cast<double*>(input.memptr()),
        reinterpret_cast<fftw_complex*>(output.memptr()),
        fftw_flag);

    assert(plan != NULL);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    fftw_cleanup_threads();
#endif
}

void generate_hermitian_matrix_from_nonredundant(arma::Mat<cx_real_t>& matrix)
{
    // R2C FFT only returns n/2+1 elements of the last dimension
    // The missing elements can be obtained by computing the conjugate of the returned elements:
    matrix.resize((matrix.n_rows - 1) * 2, matrix.n_cols);
    // First column
    for (size_t i = matrix.n_rows / 2 + 1; i < matrix.n_rows; ++i) {
        matrix.at(i, 0) = std::conj(matrix.at(matrix.n_rows - i, 0));
    }

    // Cols: 1 to n_cols
    tbb::parallel_for(tbb::blocked_range<size_t>(1, matrix.n_cols), [&matrix](const tbb::blocked_range<size_t>& r) {
        for (size_t j = r.begin(); j < r.end(); ++j) {
            for (size_t i = matrix.n_rows / 2 + 1; i < matrix.n_rows; ++i) {
                matrix.at(i, j) = std::conj(matrix.at(matrix.n_rows - i, matrix.n_cols - j));
            }
        }
    });
}
}
