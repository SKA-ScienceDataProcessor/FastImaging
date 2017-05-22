#ifndef IMAGER_H
#define IMAGER_H

// STP library includes
#include "../common/fft.h"
#include "../gridder/gridder.h"
#include "../types.h"
#include <cblas.h>

#define arc_sec_to_rad(value) ((value / 3600.0) * (M_PI / 180.0))

namespace stp {

// Image and beam data indices in std::pair structure
const uint image_slice = 0;
const uint beam_slice = 1;

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
 * @param[in] r_fft (fft_routine): Selects FFT routine to be used.
 * @param[in] image_wisdom_filename (string): FFTW wisdom filename for the image matrix (c2c fft).
 * @param[in] beam_wisdom_filename (string): FFTW wisdom filename for the beam matrix (r2c fft).
 * @param[in] gen_fullbeam (bool): If true, then generate the full hermitian matrix for beam.
 *
 * @return (std::pair<arma::cx_mat, arma::cx_mat>): Two matrices representing the generated image map and beam model (image, beam).
 */
template <typename T>
std::pair<arma::Mat<cx_real_t>, arma::Mat<cx_real_t> > image_visibilities(
    const T kernel_creator,
    const arma::cx_mat vis,
    const arma::mat uvw_lambda,
    int image_size,
    double cell_size,
    int kernel_support,
    bool kernel_exact = true,
    int oversampling = 1,
    bool normalize = true,
    fft_routine r_fft = FFTW_ESTIMATE_FFT,
    std::string image_wisdom_filename = std::string(),
    std::string beam_wisdom_filename = std::string(),
    bool gen_fullbeam = true)
{
    assert(kernel_exact || (oversampling >= 1)); // If kernel exact is false, then oversampling must be >= 1
    assert(image_size > 0);
    assert(kernel_support > 0);
    assert(cell_size > 0.0);

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cell_size) * double(image_size)));
    arma::mat uv_in_pixels = (-1.0) * uvw_lambda / grid_pixel_width_lambda; // Invert uv_in_pixels, so that forward FFT can be used then

    // Remove W column
    uv_in_pixels.shed_col(2);

    // Perform convolutional gridding of complex visibilities
    auto gridded_data = convolve_to_grid(kernel_creator, kernel_support, image_size, uv_in_pixels, vis, kernel_exact, oversampling);

    arma::Mat<cx_real_t> fft_result_image;
    arma::Mat<cx_real_t> fft_result_beam;

    // Run FFT over convolved matrices (use forward FFT since uv_in_pixels has been previously reversed: X(t)=x(-f) )
    // First: FFT of image matrix
    if (kernel_exact) {
        // Shift image if kernel_exact is true
        fftshift(gridded_data.first, false);
    }
    if (r_fft == ARMADILLO_FFT) {
        fft_result_image = arma::fft(std::move(static_cast<arma::Mat<cx_real_t> >(gridded_data.first)));
    } else {
        fft_result_image = fft_fftw(gridded_data.first, false, r_fft, image_wisdom_filename);
    }
    // Destroy gridded image matrix
    gridded_data.first.~MatStp<cx_real_t>();
    // Shift only image matrix
    fftshift(fft_result_image);

    // Second: FFT of beam matrix
    if (kernel_exact) {
        // Shift beam if kernel_exact is true
        fftshift(gridded_data.second, false);
    }
    if (r_fft == ARMADILLO_FFT) {
        fft_result_beam = arma::fft(std::move(static_cast<arma::Mat<real_t> >(gridded_data.second)));
        fftshift(fft_result_beam);
    } else {
        fft_result_beam = fft_fftw_r2c(gridded_data.second, r_fft, beam_wisdom_filename);
        if (gen_fullbeam == true) {
            generate_hermitian_matrix_from_nonredundant(fft_result_beam);
            fftshift(fft_result_beam);
        }
    }
    // Destroy gridded beam matrix
    gridded_data.second.~MatStp<real_t>();

    // Normalize image and beam such that the beam peaks at a value of 1.0 Jansky.
    if (normalize == true) {
#ifdef USE_FLOAT
        size_t beam_max_idx = cblas_isamax(fft_result_beam.n_elem, reinterpret_cast<real_t*>(fft_result_beam.memptr()), 2); // Use argument incx=2, so that only real numbers are checked
#else
        size_t beam_max_idx = cblas_idamax(fft_result_beam.n_elem, reinterpret_cast<real_t*>(fft_result_beam.memptr()), 2);
#endif
        double beam_max = fft_result_beam.at(beam_max_idx).real();
#ifdef USE_FLOAT
        // Use cblas for better performance
        cblas_csscal(fft_result_beam.n_elem, 1.0 / beam_max, reinterpret_cast<real_t*>(fft_result_beam.memptr()), 1); // Equivalent to: fft_result_beam /= beam_max;
        cblas_csscal(fft_result_image.n_elem, 1.0 / beam_max, reinterpret_cast<real_t*>(fft_result_image.memptr()), 1); // Equivalent to: fft_result_image /= beam_max;
#else
        // Use cblas for better performance
        cblas_zdscal(fft_result_beam.n_elem, 1.0 / beam_max, reinterpret_cast<real_t*>(fft_result_beam.memptr()), 1); // Equivalent to: fft_result_beam /= beam_max;
        cblas_zdscal(fft_result_image.n_elem, 1.0 / beam_max, reinterpret_cast<real_t*>(fft_result_image.memptr()), 1); // Equivalent to: fft_result_image /= beam_max;
#endif
    } else {
        // FFTW computes an unnormalized transform.
        // In order to match Numpy's inverse FFT, the result must
        // be divided by the number of elements in the matrix.
        if (r_fft != ARMADILLO_FFT) {
#ifdef USE_FLOAT
            // Use cblas for better performance
            cblas_csscal(fft_result_beam.n_elem, 1.0 / (double)(fft_result_beam.n_cols * fft_result_beam.n_rows), reinterpret_cast<real_t*>(fft_result_beam.memptr()), 1); // in-place division by (fft_result_beam.n_cols * fft_result_beam.n_rows)
            cblas_csscal(fft_result_image.n_elem, 1.0 / (double)(fft_result_image.n_cols * fft_result_image.n_rows), reinterpret_cast<real_t*>(fft_result_image.memptr()), 1); // in-place division by (fft_result_image.n_cols * fft_result_image.n_rows)
#else
            // Use cblas for better performance
            cblas_zdscal(fft_result_beam.n_elem, 1.0 / (double)(fft_result_beam.n_cols * fft_result_beam.n_rows), reinterpret_cast<real_t*>(fft_result_beam.memptr()), 1); // in-place division by (fft_result_beam.n_cols * fft_result_beam.n_rows)
            cblas_zdscal(fft_result_image.n_elem, 1.0 / (double)(fft_result_image.n_cols * fft_result_image.n_rows), reinterpret_cast<real_t*>(fft_result_image.memptr()), 1); // in-place division by (fft_result_image.n_cols * fft_result_image.n_rows)
#endif
        }
    }

    return std::make_pair(std::move(fft_result_image), std::move(fft_result_beam));
}
}
#endif /* IMAGER_H */
