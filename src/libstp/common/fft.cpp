#include "fft.h"
#include <fftw3.h>

arma::cx_mat fft_fftw(arma::cx_mat& input, bool inverse)
{
    arma::cx_mat output(arma::size(input));

    int sign = (inverse == false) ? FFTW_FORWARD : FFTW_BACKWARD;
    fftw_plan plan = fftw_plan_dft_2d(
        input.n_cols, // FFTW uses row-major order, requiring the plan
        input.n_rows, // to be passed the dimensions in reverse.
        reinterpret_cast<fftw_complex*>(input.memptr()),
        reinterpret_cast<fftw_complex*>(output.memptr()),
        sign,
        FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    // FFTW computes an unnormalized transform.
    // In order to match Numpy's inverse FFT, the result must
    // be divided by the number of elements in the matrix.
    if (inverse == true) {
        output /= input.n_cols * input.n_rows; // in-place division
    }

    return output;
}

arma::cx_mat fft_arma(arma::cx_mat& input, bool inverse)
{
    arma::cx_mat output(arma::size(input));

    if (inverse == false) {
        output = arma::fft2(input);
    } else {
        output = arma::ifft2(input);
    }

    // Armadillo normalises the FFT, so there's no
    // need for any division.
    return output;
}

arma::cx_mat fftshift(const arma::cx_mat& m, bool is_forward)
{
    arma::cx_mat result(m);
    arma::sword direction = (is_forward == true) ? -1 : 1;

    // Shift rows
    if (m.n_rows > 1) {
        arma::uword yy = arma::uword(ceil(m.n_rows / 2.0));
        result = arma::shift(result, direction * yy, 0);
    }

    // Shift columns
    if (m.n_cols > 1) {
        arma::uword xx = arma::uword(ceil(m.n_cols / 2.0));
        result = arma::shift(result, direction * xx, 1);
    }

    return result;
}
