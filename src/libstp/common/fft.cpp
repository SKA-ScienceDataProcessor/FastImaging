#include "fft.h"
#include <fftw3.h>

arma::cx_mat fft_fftw(arma::cx_mat& in, bool inverse)
{
    arma::cx_mat out(arma::size(in));

    int sign = (inverse == false) ? FFTW_FORWARD : FFTW_BACKWARD;
    fftw_plan plan = fftw_plan_dft_2d(
        in.n_cols, // FFTW uses row-major order, requiring the plan
        in.n_rows, // to be passed the dimensions in reverse.
        reinterpret_cast<fftw_complex*>(in.memptr()),
        reinterpret_cast<fftw_complex*>(out.memptr()),
        sign,
        FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    // FFTW computes an unnormalized transform.
    // In order to match Numpy's inverse FFT, the result must
    // be divided by the number of elements in the matrix.
    if (inverse == true) {
        out /= in.n_cols * in.n_rows; // in-place division
    }

    return out;
}

arma::cx_mat fft_arma(arma::cx_mat& in, bool inverse)
{
    arma::cx_mat out(arma::size(in));

    if (inverse == false) {
        out = arma::fft2(in);
    } else {
        out = arma::ifft2(in);
    }

    // Armadillo normalises the FFT, so there's no
    // need for any division.
    return out;
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

