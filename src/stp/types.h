/**
 * @file types.h
 * @brief Type definitions, global variables and enumerations.
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

namespace stp {

#ifdef USE_FLOAT
const real_t fptolerance = 1.0e-5;
#else
const real_t fptolerance = 1.0e-10;
#endif

/**
 * @brief Enum of available kernel functions
 */
enum struct KernelFunction {
    TopHat,
    Triangle,
    Sinc,
    Gaussian,
    GaussianSinc
};

/**
 * @brief Enum of available FFT algorithms
 */
enum struct FFTRoutine {
    FFTW_ESTIMATE_FFT,
    FFTW_MEASURE_FFT,
    FFTW_PATIENT_FFT,
    FFTW_WISDOM_FFT,
    FFTW_WISDOM_INPLACE_FFT, // This option is not supported (source find fails)
};

/**
 * @brief Enum of available differentiation methods used by ceres library for gaussian fitting.
 */
enum struct CeresDiffMethod {
    AutoDiff,
    AutoDiff_SingleResBlk,
    AnalyticDiff,
    AnalyticDiff_SingleResBlk,
};

/**
 * @brief Enum of available solver types used by ceres library for gaussian fitting.
 */
enum struct CeresSolverType {
    LinearSearch_BFGS,
    LinearSearch_LBFGS,
    TrustRegion_DenseQR,
};

} // stp namespace

#endif /* TYPES_H */
