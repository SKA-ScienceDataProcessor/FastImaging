#ifndef PIPELINE_H
#define PIPELINE_H

//Libstp includes
#include "../common/fft.h"
#include "../gridder/gridder.h"

// CMake variable
#ifndef _ARMA
#define _ARMA 0
#endif

constexpr double
arc_sec_to_rad(double value)
{
    return ((value / 3600.0) * (M_PI / 180.0));
}

/**
 * @brief image_visibilities function
 *
 * This function is a first version adapted from the function in imager.py file.
 *
 * @param[in] vis (cx_mat) :
 * @param[in] uvw_lambda (cx_mat) :
 * @param[in] image_size (int) :
 * @param[in] cell_size (double) :
 *
 * @return result of the fft on the mat inputed
 */
template <typename T>
arma::cx_cube image_visibilities(arma::cx_mat vis, arma::cx_mat uvw_lambda, int image_size, double cell_size, const double support, const double oversampling, const bool pad, const bool normalize, const T& kernel_creator)
{
    // Size of a UV-grid pixel, in multiples of wavelength (lambda):

    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cell_size) * image_size));
    arma::mat uvw_in_pixels = arma::real(uvw_lambda) / grid_pixel_width_lambda;

    arma::mat uv_in_pixels(10, 2);
    uv_in_pixels.col(0) = uvw_in_pixels.col(0);
    uv_in_pixels.col(1) = uvw_in_pixels.col(1);

    arma::cx_cube result = convolve_to_grid(support, image_size, uv_in_pixels, vis, oversampling, pad, normalize, kernel_creator);
    arma::cx_cube fft_result(image_size, image_size, 2);

    arma::cx_mat shifted_image = fftshift(result.slice(0), false);
    arma::cx_mat shifted_beam = fftshift(result.slice(1), false);

    if (_ARMA) {
        // Armadillo implementation
        fft_result.slice(0) = fftshift(fft_arma(shifted_image, true));
        fft_result.slice(1) = fftshift(fft_arma(shifted_beam, true));
    } else {
        // FFTW implementation
        fft_result.slice(0) = fftshift(fft_fftw(shifted_image, true));
        fft_result.slice(1) = fftshift(fft_fftw(shifted_beam, true));
    }

    return fft_result;
}
#endif /* PIPELINE_H */
