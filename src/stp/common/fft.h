/**
* @file fft.h
* @brief Function prototypes of FFT.
*/

#ifndef FFT_H
#define FFT_H

#include "../types.h"
#include "matrix_math.h"
#include <armadillo>

namespace stp {

/**
 * @brief Init FFTW threads and import FFTW wisdom file if required.
 *
 * @param[in] r_fft (FFTRoutine enum) : FFT routine to be used: defines the FFTW planner flag
 * @param[in] fft_wisdom_filename (string): FFTW wisdom filename for FFT execution.
 */
void init_fftw(FFTRoutine r_fft, std::string fft_wisdom_filename);

/**
 * @brief Performs the backward fast fourier transform of a halfplane complex matrix using the FFTW library (complex to real FFT)
 *
 * Receives the halfplane complex matrix (n_rows = n_cols/2 +1) and performs the backward fast fourier transform returning a real
 * output matrix. Find more details about c2r (complex to real) FFT in the FFTW manual.
 *
 * @param[in] input (arma::Mat) : Complex input matrix to be transformed using fft
 * @param[in] output (arma::Mat) : Real output matrix with the fft result
 * @param[in] r_fft (FFTRoutine enum) : FFT routine to be used: defines the FFTW planner flag
 */
void fft_fftw_c2r(arma::Mat<cx_real_t>& input, arma::Mat<real_t>& output, FFTRoutine r_fft = FFTRoutine::FFTW_ESTIMATE_FFT);

/**
 * @brief Performs the forward fast fourier transform of a real matrix using the FFTW library (real to complex FFT)
 *
 * Receives the real matrix (n_rows = n_cols) and performs the forward fast fourier transform returning a complex
 * output matrix. Find more details about r2c (real to complex) FFT in the FFTW manual.
 *
 * @param[in] input (arma::Mat) : Real input matrix to be transformed using fft
 * @param[in] output (arma::Mat) : Complex halfplane output matrix with the fft result
 * @param[in] r_fft (FFTRoutine enum) : FFT routine to be used: defines the FFTW planner flag
 */
void fft_fftw_r2c(arma::Mat<real_t>& input, arma::Mat<cx_real_t>& output, FFTRoutine r_fft = FFTRoutine::FFTW_ESTIMATE_FFT);

/**
 * @brief Performs the forward fast fourier transform of a complex matrix using the FFTW library (complex to complex FFT)
 *
 * Receives the complex matrix (n_rows = n_cols) and performs the forward fast fourier transform returning a complex
 * output matrix. Find more details about c2c (complex to complex) FFT in the FFTW manual.
 *
 * @param[in] input (arma::Mat) : Complex input matrix to be transformed using fft
 * @param[in] output (arma::Mat) : Complex halfplane output matrix with the fft result
 * @param[in] r_fft (FFTRoutine enum) : FFT routine to be used: defines the FFTW planner flag
 */
void fft_fftw_c2c(arma::Mat<cx_real_t>& input, arma::Mat<cx_real_t>& output, FFTRoutine r_fft, bool forward = true);

/**
 * @brief Performs the forward fast fourier transform of a real vector (1D) using the FFTW library (real to complex FFT)
 *
 * Receives the real vector and performs the forward fast fourier transform returning a complex output vector in halfcomplex
 * format as specified by FFTW_R2HC transform kind (first all real values and then all imaginary values).
 * This allows the output and input arrays having the same size. Find more details about r2r (real to real) FFT in the FFTW manual.
 *
 * @param[in] input (arma::Mat) : Real input vector to be transformed using fft
 * @param[in] output (arma::Mat) : Real halfplane output vector with the fft result
 * @param[in] r_fft (FFTRoutine enum) : FFT routine to be used: defines the FFTW planner flag
 */
void fft_fftw_dft_r2r_1d(arma::Col<real_t>& input, arma::Col<real_t>& output, FFTRoutine r_fft);

void fft_fftw_dft_c2c_1d(arma::Col<cx_real_t>& input, arma::Col<cx_real_t>& output, FFTRoutine r_fft);

/**
 * @brief Generates a hermitian matrix from the non-redundant values
 *
 * Receives a complex matrix with the non-redundant values (as returned by the fft_r2c FFTW function)
 * which is used to generate the full hermitian matrix.
 *
 * @param[in] matrix (arma::Mat) : Complex matrix with the non-redundant values
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
        arma::uword yy = arma::uword(floor(m.n_rows / 2.0));
        m = matrix_shift(m, direction * yy, 0);
    }

    // Shift columns
    if (m.n_cols > 1) {
        arma::uword xx = arma::uword(floor(m.n_cols / 2.0));
        m = matrix_shift(m, direction * xx, 1);
    }
}
}

#endif /* FFT_H */
