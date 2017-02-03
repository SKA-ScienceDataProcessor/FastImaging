#ifndef FFT_H
#define FFT_H

#include "vector_math.h"
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
