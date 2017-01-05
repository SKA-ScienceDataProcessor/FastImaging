#ifndef FFT_H
#define FFT_H

#include <armadillo>

namespace stp {

/**
 * @brief Performs a fast fourier transform on a matrix using the armadillo library
 *
 * Receives a complex matrix and performs a fast fourier transform or an inverse fast fourier transform
 * if the parameter inverse is false
 *
 * @param[in] in (cx_mat) : Complex matrix to perform the fft
 * @param[in] inverse (bool) : Boolean flag to indicate if we perform an inverse transform
 *
 * @return The transformed matrix
 */
arma::cx_mat fft_arma(arma::cx_mat& input, bool inverse = false);

/**
 * @brief Performs a fast fourier transform on a matrix using the FFTW library
 *
 * Receives a complex matrix and performs a fast fourier transform or an inverse fast fourier transform
 * if the parameter inverse is false
 *
 * @param[in] m (cx_mat) : Complex matrix to perform the fft
 * @param[in] inverse (bool) : Boolean flag to indicate if we perform an inverse transform
 *
 * @return The transformed matrix
 */
arma::cx_mat fft_fftw(arma::cx_mat& m, bool inverse = false);

/**
 * @brief fftshift
 *
 * Shift the zero-frequency component to the centre of the spectrum.
 *
 * @param[in] m (cx_mat) : The matrix to be shifted
 * @param[in] is_forward (bool) Shifts forward if true (default), backward otherwise
 * @return The shifted matrix
 */
arma::cx_mat fftshift(const arma::cx_mat& m, bool is_forward = true);
}

#endif /* FFT_H */
