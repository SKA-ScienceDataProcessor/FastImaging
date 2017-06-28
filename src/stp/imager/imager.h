#ifndef IMAGER_H
#define IMAGER_H

// STP library includes
#include "../common/fft.h"
#include "../gridder/gridder.h"
#include "../types.h"
#include <cblas.h>

#define arc_sec_to_rad(value) ((value / 3600.0) * (M_PI / 180.0))
#define NUM_TIME_INST 10

namespace stp {

#ifdef FUNCTION_TIMINGS
extern std::vector<std::chrono::high_resolution_clock::time_point> times_iv;
#endif

// Image and beam data indices in std::pair structure
const uint image_slice = 0;
const uint beam_slice = 1;

/**
 * @brief Convert the input visibilities to an array of half-plane visibilities.
 *
 * @param[in] vis (arma::cx_mat): Complex visibilities to be converted (1D array).
 * @param[in] uvw_lambda (arma::mat): UVW-coordinates of complex visibilities to be converted.
 *                                    2D double array with 3 columns. Assumed ordering is u,v,w.
 * @param[in] kernel_support (int): Kernel support radius.
 */
void convert_to_halfplane_visibilities(arma::mat& uv_in_pixels, arma::cx_mat& vis, int kernel_support);

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
 * @param[in] r_fft (FFTRoutine): Selects FFT routine to be used.
 * @param[in] image_wisdom_filename (string): FFTW wisdom filename for the image matrix (c2c fft).
 * @param[in] beam_wisdom_filename (string): FFTW wisdom filename for the beam matrix (r2c fft).
 *
 * @return (std::pair<arma::mat, arma::mat>): Two matrices representing the generated image map and beam model (image, beam).
 */
template <typename T>
std::pair<arma::Mat<real_t>, arma::Mat<real_t>> image_visibilities(
    const T kernel_creator,
    arma::cx_mat vis,
    arma::mat uvw_lambda,
    int image_size,
    double cell_size,
    int kernel_support,
    bool kernel_exact = true,
    int oversampling = 1,
    bool normalize = true,
    FFTRoutine r_fft = FFTW_ESTIMATE_FFT,
    std::string image_wisdom_filename = std::string(),
    std::string beam_wisdom_filename = std::string())
{
    assert(kernel_exact || (oversampling >= 1)); // If kernel exact is false, then oversampling must be >= 1
    assert(image_size > 0);
    assert(kernel_support > 0);
    assert(cell_size > 0.0);

#ifdef FUNCTION_TIMINGS
    times_iv.reserve(NUM_TIME_INST);
    times_iv.push_back(std::chrono::high_resolution_clock::now());
#endif

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cell_size) * double(image_size)));
    arma::mat uv_in_pixels = (uvw_lambda / grid_pixel_width_lambda);

    // Remove W column
    uv_in_pixels.shed_col(2);

    // If a visibility point is located in the top half-plane, move it to the bottom half-plane to a symmetric position with respect to the matrix centre (0,0)
    convert_to_halfplane_visibilities(uv_in_pixels, vis, kernel_support);

    // Perform convolutional gridding of complex visibilities
    std::pair<MatStp<cx_real_t>, MatStp<cx_real_t>> gridded_data = convolve_to_grid(kernel_creator, kernel_support, image_size, uv_in_pixels, vis, kernel_exact, oversampling);

#ifdef FUNCTION_TIMINGS
    times_iv.push_back(std::chrono::high_resolution_clock::now());
#endif

    arma::Mat<real_t> fft_result_image;
    arma::Mat<real_t> fft_result_beam;

    // Reuse gridded_data buffer if FFT is INPLACE
    if (r_fft == stp::FFTW_WISDOM_INPLACE_FFT) {
        fft_result_image = std::move(arma::Mat<real_t>(reinterpret_cast<real_t*>(gridded_data.first.memptr()), (gridded_data.first.n_rows) * 2, gridded_data.first.n_cols, false, false));
        fft_result_beam = std::move(arma::Mat<real_t>(reinterpret_cast<real_t*>(gridded_data.second.memptr()), (gridded_data.second.n_rows) * 2, gridded_data.second.n_cols, false, false));
    }

    // Run FFT over convolved matrices (use forward FFT since uv_in_pixels has been previously reversed: X(t)=x(-f) )
    // First: FFT of image matrix
    //if (kernel_exact) {
    // Shift image if kernel_exact is true
    //fftshift(gridded_data.first, false);
    //}
    fft_fftw_c2r(gridded_data.first, fft_result_image, r_fft, image_wisdom_filename);

    // Delete gridded image matrix (only if FFT is not inplace)
    if (r_fft != stp::FFTW_WISDOM_INPLACE_FFT) {
        gridded_data.first.delete_matrix_buffer();
    }
    // Shift image matrix (beam does not need to be shifted)
    //fftshift(fft_result_image);

    // Second: FFT of beam matrix
    //if (kernel_exact) {
    // Shift beam if kernel_exact is true
    //fftshift(gridded_data.second, false);
    //}
    fft_fftw_c2r(gridded_data.second, fft_result_beam, r_fft, beam_wisdom_filename);
    // Delete gridded beam matrix (only if FFT is not inplace)
    if (r_fft != stp::FFTW_WISDOM_INPLACE_FFT) {
        gridded_data.second.delete_matrix_buffer();
    }

#ifdef FUNCTION_TIMINGS
    times_iv.push_back(std::chrono::high_resolution_clock::now());
#endif
    // Normalize image and beam such that the beam peaks at a value of 1.0 Jansky.
    if (normalize == true) {
#ifdef USE_FLOAT
        size_t beam_max_idx = cblas_isamax(fft_result_beam.n_elem, fft_result_beam.memptr(), 1);
#else
        size_t beam_max_idx = cblas_idamax(fft_result_beam.n_elem, fft_result_beam.memptr(), 1);
#endif
        real_t beam_max = fft_result_beam.at(beam_max_idx);
        assert(std::abs(beam_max) > 0.0);
#ifdef USE_FLOAT
        // Use cblas for better performance
        cblas_sscal(fft_result_beam.n_elem, 1.0f / beam_max, fft_result_beam.memptr(), 1); // Equivalent to: fft_result_beam /= beam_max;
        cblas_sscal(fft_result_image.n_elem, 1.0f / beam_max, fft_result_image.memptr(), 1); // Equivalent to: fft_result_image /= beam_max;
#else
        // Use cblas for better performance
        cblas_dscal(fft_result_beam.n_elem, 1.0 / beam_max, fft_result_beam.memptr(), 1); // Equivalent to: fft_result_beam /= beam_max;
        cblas_dscal(fft_result_image.n_elem, 1.0 / beam_max, fft_result_image.memptr(), 1); // Equivalent to: fft_result_image /= beam_max;
#endif
    }

#ifdef FUNCTION_TIMINGS
    times_iv.push_back(std::chrono::high_resolution_clock::now());
#endif

    return std::make_pair(std::move(fft_result_image), std::move(fft_result_beam));
}
}
#endif /* IMAGER_H */
