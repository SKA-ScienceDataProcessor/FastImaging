/**
* @file fft.cpp
* @brief Implementation of FFT functions.
*/

#include "fft.h"
#include <cassert>
#include <fftw3.h>
#include <thread>

namespace stp {

void init_fftw(FFTRoutine r_fft, std::string fft_wisdom_filename)
{
// Init fftw threads
#ifdef USE_FLOAT
    if (!fftwf_init_threads()) {
        throw std::runtime_error("Failed to init FFTW threads");
        assert(0);
    }
    fftwf_plan_with_nthreads(std::thread::hardware_concurrency());
#else
    if (!fftw_init_threads()) {
        throw std::runtime_error("Failed to init FFTW threads");
        assert(0);
    }
    fftw_plan_with_nthreads(std::thread::hardware_concurrency());
#endif

    // Import Wisdom file
    if ((r_fft == FFTRoutine::FFTW_WISDOM_FFT) || (r_fft == FFTRoutine::FFTW_WISDOM_INPLACE_FFT)) {
#ifdef USE_FLOAT
        if (!fftwf_import_wisdom_from_filename(fft_wisdom_filename.c_str())) {
#else
        if (!fftw_import_wisdom_from_filename(fft_wisdom_filename.c_str())) {
#endif
            throw std::runtime_error("Failed to read FFTW wisdom file: " + fft_wisdom_filename);
            assert(0);
        }
    }
}

void fft_fftw_c2r(arma::Mat<cx_real_t>& input, arma::Mat<real_t>& output, FFTRoutine r_fft)
{
    size_t n_rows = (input.n_rows % 2 == 0) ? (input.n_rows * 2) : (input.n_rows - 1) * 2;
    size_t n_cols = input.n_cols;

    if (reinterpret_cast<real_t*>(input.memptr()) != output.memptr()) {
        output.set_size(n_rows, n_cols);
    }
    unsigned int fftw_flag = FFTW_ESTIMATE;

    switch (r_fft) {
    case FFTRoutine::FFTW_ESTIMATE_FFT:
        fftw_flag = FFTW_ESTIMATE;
        break;
    case FFTRoutine::FFTW_MEASURE_FFT:
        // Do not use this mode as the plan generation is very slow
        fftw_flag = FFTW_MEASURE;
        assert(0);
        break;
    case FFTRoutine::FFTW_PATIENT_FFT:
        // Do not use this mode as the plan generation is extremely slow
        fftw_flag = FFTW_PATIENT;
        assert(0);
        break;
    case FFTRoutine::FFTW_WISDOM_FFT:
    case FFTRoutine::FFTW_WISDOM_INPLACE_FFT:
        fftw_flag = FFTW_WISDOM_ONLY;
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

    if (plan == NULL) {
        throw std::runtime_error("Failed to create FFTW plan.");
    }

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
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
#endif
}

void fft_fftw_r2c(arma::Mat<real_t>& input, arma::Mat<cx_real_t>& output, FFTRoutine r_fft)
{
    size_t n_rows = input.n_rows / 2 + 1;
    size_t n_cols = input.n_cols;
    if (input.memptr() != reinterpret_cast<real_t*>(output.memptr())) {
        output.set_size(n_rows, n_cols);
    }
    unsigned int fftw_flag = FFTW_ESTIMATE;

    switch (r_fft) {
    case FFTRoutine::FFTW_ESTIMATE_FFT:
        fftw_flag = FFTW_ESTIMATE;
        break;
    case FFTRoutine::FFTW_MEASURE_FFT:
        // Do not use this mode as the plan generation is very slow
        fftw_flag = FFTW_MEASURE;
        assert(0);
        break;
    case FFTRoutine::FFTW_PATIENT_FFT:
        // Do not use this mode as the plan generation is extremely slow
        fftw_flag = FFTW_PATIENT;
        assert(0);
        break;
    case FFTRoutine::FFTW_WISDOM_FFT:
    case FFTRoutine::FFTW_WISDOM_INPLACE_FFT:
        fftw_flag = FFTW_WISDOM_ONLY;
        break;
    default:
        assert(0);
        break;
    }

#ifdef USE_FLOAT
    fftwf_plan plan = fftwf_plan_dft_r2c_2d(
        input.n_cols, // FFTW uses row-major order, requiring the plan
        input.n_rows, // to be passed the dimensions in reverse.
        reinterpret_cast<float*>(input.memptr()),
        reinterpret_cast<fftwf_complex*>(output.memptr()),
        fftw_flag);

    assert(plan != NULL);

    if (plan == NULL) {
        throw std::runtime_error("Failed to create FFTW plan.");
    }

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
#else
    fftw_plan plan = fftw_plan_dft_r2c_2d(
        input.n_cols, // FFTW uses row-major order, requiring the plan
        input.n_rows, // to be passed the dimensions in reverse.
        reinterpret_cast<double*>(input.memptr()),
        reinterpret_cast<fftw_complex*>(output.memptr()),
        fftw_flag);

    assert(plan != NULL);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
#endif
}

void fft_fftw_c2c(arma::Mat<cx_real_t>& input, arma::Mat<cx_real_t>& output, FFTRoutine r_fft, bool forward)
{
    size_t n_rows = input.n_rows;
    size_t n_cols = input.n_cols;
    if (input.memptr() != output.memptr()) {
        output.set_size(n_rows, n_cols);
    }
    unsigned int fftw_flag = FFTW_ESTIMATE;

    switch (r_fft) {
    case FFTRoutine::FFTW_ESTIMATE_FFT:
        fftw_flag = FFTW_ESTIMATE;
        break;
    case FFTRoutine::FFTW_MEASURE_FFT:
        // Do not use this mode as the plan generation is very slow
        fftw_flag = FFTW_MEASURE;
        assert(0);
        break;
    case FFTRoutine::FFTW_PATIENT_FFT:
        // Do not use this mode as the plan generation is extremely slow
        fftw_flag = FFTW_PATIENT;
        assert(0);
        break;
    case FFTRoutine::FFTW_WISDOM_FFT:
    case FFTRoutine::FFTW_WISDOM_INPLACE_FFT:
        fftw_flag = FFTW_WISDOM_ONLY;
        break;
    default:
        assert(0);
        break;
    }

    int direction = FFTW_FORWARD;
    if (!forward) {
        direction = FFTW_BACKWARD;
    }

#ifdef USE_FLOAT
    fftwf_plan plan = fftwf_plan_dft_2d(
        n_rows, // FFTW uses row-major order, requiring the plan
        n_cols, // to be passed the dimensions in reverse.
        reinterpret_cast<fftwf_complex*>(input.memptr()),
        reinterpret_cast<fftwf_complex*>(output.memptr()),
        direction,
        fftw_flag);

    assert(plan != NULL);

    if (plan == NULL) {
        throw std::runtime_error("Failed to create FFTW plan.");
    }

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
#else
    fftw_plan plan = fftw_plan_dft_2d(
        n_rows, // FFTW uses row-major order, requiring the plan
        n_cols, // to be passed the dimensions in reverse.
        reinterpret_cast<fftw_complex*>(input.memptr()),
        reinterpret_cast<fftw_complex*>(output.memptr()),
        direction,
        fftw_flag);

    assert(plan != NULL);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
#endif
}

void fft_fftw_dft_r2r_1d(arma::Col<real_t>& input, arma::Col<real_t>& output, FFTRoutine r_fft)
{
    size_t n_elems = input.size();

    if (input.memptr() != output.memptr()) {
        output.set_size(arma::size(input));
    }

    unsigned int fftw_flag = FFTW_ESTIMATE;

    switch (r_fft) {
    case FFTRoutine::FFTW_ESTIMATE_FFT:
        fftw_flag = FFTW_ESTIMATE;
        break;
    case FFTRoutine::FFTW_MEASURE_FFT:
        // Do not use this mode as the plan generation is very slow
        fftw_flag = FFTW_MEASURE;
        assert(0);
        break;
    case FFTRoutine::FFTW_PATIENT_FFT:
        // Do not use this mode as the plan generation is extremely slow
        fftw_flag = FFTW_PATIENT;
        assert(0);
        break;
    case FFTRoutine::FFTW_WISDOM_FFT:
    case FFTRoutine::FFTW_WISDOM_INPLACE_FFT:
        fftw_flag = FFTW_WISDOM_ONLY;
        break;
    default:
        assert(0);
        break;
    }

#ifdef USE_FLOAT
    fftwf_plan plan
        = fftwf_plan_r2r_1d(
            n_elems,
            reinterpret_cast<float*>(input.memptr()),
            reinterpret_cast<float*>(output.memptr()),
            FFTW_R2HC,
            fftw_flag);

    assert(plan != NULL);

    if (plan == NULL) {
        throw std::runtime_error("Failed to create FFTW plan.");
    }

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
#else
    fftw_plan plan
        = fftw_plan_r2r_1d(
            n_elems,
            reinterpret_cast<double*>(input.memptr()),
            reinterpret_cast<double*>(output.memptr()),
            FFTW_R2HC,
            fftw_flag);

    assert(plan != NULL);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
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
    tbb::parallel_for(tbb::blocked_range<size_t>(1, matrix.n_cols), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t j = r.begin(); j < r.end(); ++j) {
            for (size_t i = matrix.n_rows / 2 + 1; i < matrix.n_rows; ++i) {
                matrix.at(i, j) = std::conj(matrix.at(matrix.n_rows - i, matrix.n_cols - j));
            }
        }
    });
}
}
