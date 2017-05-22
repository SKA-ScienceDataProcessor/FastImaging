#ifndef FFT_H
#define FFT_H

#include "vector_math.h"
#include <armadillo>

namespace stp {

// Available FFT algorithms
typedef enum {
    FFTW_ESTIMATE_FFT,
    FFTW_MEASURE_FFT,
    FFTW_PATIENT_FFT,
    FFTW_WISDOM_FFT,
    ARMADILLO_FFT
} fft_routine;

/**
 * @brief Performs a fast fourier transform on a complex matrix using the FFTW library
 *
 * Receives a complex matrix and performs a fast fourier transform or an inverse fast fourier transform
 * if the parameter inverse is false
 *
 * @param[in] m (cx_mat) : Complex matrix to perform the fft
 * @param[in] inverse (bool) : Boolean flag to indicate if we perform an inverse transform
 * @param[in] r_fft (fft_routine enum) : Indicates FFTW planner flag to be used based on input fft routine config
 * @param[in] wisdom_filename (string) : Input filename of FFTW wisdom (optional)
 *
 * @return (cx_mat) The transformed matrix
 */
arma::Mat<cx_real_t> fft_fftw(arma::Mat<cx_real_t>& m, bool inverse = false, fft_routine r_fft = FFTW_ESTIMATE_FFT, const std::string& wisdom_filename = std::string());

/**
 * @brief Performs a fast fourier transform on a real matrix using the FFTW library (real to complex FFT)
 *
 * Receives a real matrix and performs a fast fourier transform (only performs forward FFT)
 *
 * @param[in] m (mat) : Real matrix to perform the fft
 * @param[in] r_fft (fft_routine enum) : Indicates FFTW planner flag to be used based on input fft routine config
 * @param[in] wisdom_filename (string) : Input filename of FFTW wisdom (optional)
 *
 * @return (cx_mat) The transformed matrix
 */
arma::Mat<cx_real_t> fft_fftw_r2c(arma::Mat<real_t>& m, fft_routine r_fft = FFTW_ESTIMATE_FFT, const std::string& wisdom_filename = std::string());

/**
 * @brief Generates a hermitian matrix from the non-redundant values
 *
 * Receives a complex matrix with the non-redundant values (as returned by the fft_r2c FFTW function)
 * which is used to generate the full hermitian matrix.
 *
 * @param[in] matrix (cx_mat) : Matrix with the non-redundant values
 *
 * @return (cx_mat) The hermitian matrix
 */
void generate_hermitian_matrix_from_nonredundant(arma::Mat<cx_real_t>& matrix);

/**
 * @brief Performs matrix circular shift as needed for iFFT
 *
 * Shift the zero-frequency component to the centre of the spectrum.
 *
 * @param[in] m (arma::Mat<T>) : The matrix to be shifted (shifted matrix is also stored here)
 * @param[in] is_forward (bool) Shifts forward if true (default), backward otherwise
 */
template <typename T>
void fftshift(arma::Mat<T>& m, bool is_forward = true)
{
    arma::sword direction = (is_forward == true) ? -1 : 1;

    // Shift rows
    if (m.n_rows > 1) {
        arma::uword yy = arma::uword(ceil(m.n_rows / 2.0));
        m = matrix_shift(m, direction * yy, 0);
    }

    // Shift columns
    if (m.n_cols > 1) {
        arma::uword xx = arma::uword(ceil(m.n_cols / 2.0));
        m = matrix_shift(m, direction * xx, 1);
    }
}
}

#endif /* FFT_H */
