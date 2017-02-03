#ifndef IMAGER_H
#define IMAGER_H

// STP library includes
#include "../common/fft.h"
#include "../gridder/gridder.h"
#include <cblas.h>

#define arc_sec_to_rad(value) ((value / 3600.0) * (M_PI / 180.0))

namespace stp {

// Image and beam data indices in std::pair structure
const uint image_slice = 0;
const uint beam_slice = 1;

// Available FFT algorithms
typedef enum { FFTW,
    ARMAFFT } fft_function_type;

/**
 * @brief Generates image and beam data from input visibilities.
 *
 * Performs convolutional gridding of input visibilities and applies ifft.
 * Returns two arrays representing the image map and beam model.
 *
 * @param[in] kernel_creator (typename T): Callable object that returns a convolution kernel.
 * @param[in] vis (arma::cx_mat): Complex visibilities (1D array).
 * @param[in] uvw_lambda (arma::mat): UVW-coordinates of complex visibilities. Units are multiples of wavelength.
 *                                    2D double array with 3 columns. Assumed ordering is u,v,w.
 * @param[in] image_size (int): Width of the image in pixels. Assumes (image_size/2, image_size/2) corresponds to the origin in UV-space.
 * @param[in] cell_size (double): Angular-width of a synthesized pixel in the image to be created (arcsecond).
 * @param[in] kernel_support (int): Defines the 'radius' of the bounding box within which convolution takes place (also known as half-support).
 *                                  Box width in pixels = 2*support + 1. The central pixel is the one nearest to the UV co-ordinates.
 * @param[in] kernel_exact (bool): Calculate exact kernel-values for every UV-sample.
 * @param[in] oversampling (int): Controls kernel-generation if 'kernel_exact == False'. Larger values give a finer-sampled set of pre-cached kernels.
 * @param[in] normalize (bool): Whether or not the returned image and beam should be normalized such that the beam peaks at a value of 1.0 Jansky.
 *                              You normally want this to be true, but it may be interesting to check the raw values for debugging purposes.
 * @param[in] f_fft (fft_function_type): Selects FFT implementation to be used.
 *
 * @return (std::pair<arma::cx_mat, arma::cx_mat>): Two matrices representing the generated image map and beam model (image, beam).
 */
template <typename T>
std::pair<arma::cx_mat, arma::cx_mat> image_visibilities(
    const T& kernel_creator,
    const arma::cx_mat& vis,
    const arma::mat& uvw_lambda,
    int image_size,
    double cell_size,
    int kernel_support,
    bool kernel_exact = true,
    int oversampling = 1,
    bool normalize = true,
    fft_function_type f_fft = FFTW)
{
    assert(kernel_exact || (oversampling >= 1)); // If kernel exact is false, then oversampling must be >= 1
    assert(image_size > 0);
    assert(kernel_support > 0);
    assert(cell_size > 0.0);

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cell_size) * double(image_size)));
    arma::mat uv_in_pixels = uvw_lambda / grid_pixel_width_lambda;

    // Remove W column
    uv_in_pixels.shed_col(2);

    // Perform convolutional gridding of complex visibilities
    std::pair<arma::cx_mat, arma::mat> result = convolve_to_grid(kernel_creator, kernel_support, image_size, uv_in_pixels, vis, kernel_exact, oversampling);

    arma::cx_mat fft_result_image;
    arma::cx_mat fft_result_beam;

    // Run iFFT over convolved grid
    switch (f_fft) {
    case FFTW:
        // FFTW implementation
        fftshift(result.first, false);
        fft_result_image = fft_fftw(result.first, true);
        result.first.set_size(0); // Destroy unused matrix
        fftshift(fft_result_image);

        fftshift(result.second, false);
        fft_result_beam = arma::conv_to<arma::cx_mat>::from(result.second);
        result.second.set_size(0); // Destroy unused matrix
        fft_result_beam = fft_fftw(fft_result_beam, true);
        fftshift(fft_result_beam);
        break;
    case ARMAFFT:
        // Armadillo implementation
        fftshift(result.first, false);
        fft_result_image = fft_arma(result.first, true);
        result.first.set_size(0); // Destroy unused matrix
        fftshift(fft_result_image);

        fftshift(result.second, false);
        fft_result_beam = arma::conv_to<arma::cx_mat>::from(result.second);
        result.second.set_size(0); // Destroy unused matrix
        fft_result_beam = fft_arma(fft_result_beam, true);
        fftshift(fft_result_beam);
        break;
    default:
        assert(0);
    }

    // Normalize image and beam such that the beam peaks at a value of 1.0 Jansky.
    if (normalize == true) {
        double beam_max = arma::real(fft_result_beam).max();

        // Use cblas for better performance
        int n_elem = fft_result_beam.n_elem;
        cblas_zdscal(n_elem, 1 / beam_max, fft_result_beam.memptr(), 1); // Equivalent to: fft_result_beam /= beam_max;
        n_elem = fft_result_image.n_elem;
        cblas_zdscal(n_elem, 1 / beam_max, fft_result_image.memptr(), 1); // Equivalent to: fft_result_image /= beam_max;
    }

    return std::make_pair(std::move(fft_result_image), std::move(fft_result_beam));
}
}
#endif /* IMAGER_H */
