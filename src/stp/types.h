/** @file types.h
 *  @brief Include of types
 *
 *  @bug No known bugs.
 */

#ifndef TYPES_H
#define TYPES_H

#include <complex>

// Defines single- or double- precision floating point type for STP library
#ifdef USE_FLOAT
using real_t = float;
using cx_real_t = std::complex<float>;
#else
using real_t = double;
using cx_real_t = std::complex<double>;
#endif

#ifdef USE_FLOAT
const real_t tolerance = 1.0e-5;
#else
const real_t tolerance = 1.0e-10;
#endif

namespace stp {

/**
 * @brief The KernelFunction enum
 */
enum struct KernelFunction {
    TopHat = 0,
    Triangle,
    Sinc,
    Gaussian,
    GaussianSinc
};

// Available FFT algorithms
typedef enum {
    FFTW_ESTIMATE_FFT,
    FFTW_MEASURE_FFT,
    FFTW_PATIENT_FFT,
    FFTW_WISDOM_FFT,
    FFTW_WISDOM_INPLACE_FFT, // This routine is not supported (source find will fail)
} FFTRoutine;
} // stp namespace

#endif /* TYPES_H */
