#include "fft.h"
#include <cblas.h>
#include <fftw3.h>
#include <thread>

namespace stp {

arma::cx_mat fft_fftw(arma::cx_mat& input, bool inverse)
{
    arma::cx_mat output(arma::size(input));

    fftw_init_threads();
    fftw_plan_with_nthreads(std::thread::hardware_concurrency());

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

    fftw_cleanup_threads();

    return output;
}

arma::cx_mat fft_arma(arma::cx_mat& m, bool inverse)
{
    arma::cx_mat result(arma::size(m));

    if (inverse == false) {
        result = arma::fft2(m);
    } else {
        result = arma::ifft2(m);
    }

    // Armadillo normalises the FFT, so there's no
    // need for any division.
    return result;
}
}
