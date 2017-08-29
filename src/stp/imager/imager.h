/**
 * @file imager.h
 * @brief Function prototypes of the imager.
 */

#ifndef IMAGER_H
#define IMAGER_H

// STP library includes
#include "../common/fft.h"
#include "../gridder/gridder.h"
#include "../types.h"

#define arc_sec_to_rad(value) ((value / 3600.0) * (M_PI / 180.0))
#define NUM_TIME_INST 10

namespace stp {

#ifdef FUNCTION_TIMINGS
extern std::vector<std::chrono::high_resolution_clock::time_point> times_iv;
#endif

/**
 * @brief Convert the input visibilities to an array of half-plane visibilities.
 *
 * @param[in] uvw_lambda (arma::mat): UVW-coordinates of complex visibilities to be converted.
 *                                    2D double array with 3 columns. Assumed ordering is u,v,w.
 * @param[in] vis (arma::cx_mat): Complex visibilities to be converted (1D array).
 * @param[in] vis_weights (arma::mat): Visibility weights (1D array).
 * @param[in] kernel_support (int): Kernel support radius.
 */
void convert_to_halfplane_visibilities(arma::mat& uv_in_pixels, arma::cx_mat& vis, arma::mat& vis_weights, int kernel_support);

/**
 * @brief Generates image and beam data from input visibilities.
 *
 * Performs convolutional gridding of input visibilities and applies ifft.
 * Returns two arrays representing the image map and beam model.
 *
 * @param[in] kernel_creator (typename T): Callable object that returns a convolution kernel.
 * @param[in] vis (arma::cx_mat): Complex visibilities (1D array).
 * @param[in] vis_weights (arma::mat): Visibility weights (1D array).
 * @param[in] uvw_lambda (arma::mat): UVW-coordinates of complex visibilities. Units are multiples of wavelength.
 *                                    2D double array with 3 columns. Assumed ordering is u,v,w.
 * @param[in] image_size (int): Width of the image in pixels. Assumes (image_size/2, image_size/2) corresponds to the origin in UV-space.
 * @param[in] cell_size (double): Angular-width of a synthesized pixel in the image to be created (arcsecond).
 * @param[in] kernel_support (int): Defines the 'radius' of the bounding box within which convolution takes place (also known as half-support).
 *                                  Box width in pixels = 2*support + 1. The central pixel is the one nearest to the UV co-ordinates.
 * @param[in] kernel_exact (bool): Calculate exact kernel-values for every UV-sample.
 * @param[in] oversampling (int): Controls kernel-generation if 'kernel_exact == False'. Larger values give a finer-sampled set of pre-cached kernels.
 * @param[in] generate_beam (bool): Enables generation of gridded sampling matrix. Default is false.
 * @param[in] r_fft (FFTRoutine): Selects FFT routine to be used.
 * @param[in] fft_wisdom_filename (string): FFTW wisdom filename for the image and beam (c2r fft).
 *
 * @return (std::pair<arma::mat, arma::mat>): Two matrices representing the generated image map and beam model (image, beam).
 */
template <typename T>
std::pair<arma::Mat<real_t>, arma::Mat<real_t>> image_visibilities(
    const T kernel_creator,
    const arma::cx_mat& vis,
    const arma::mat& vis_weights,
    const arma::mat& uvw_lambda,
    int image_size,
    double cell_size,
    int kernel_support,
    bool kernel_exact = true,
    int oversampling = 1,
    bool generate_beam = false,
    FFTRoutine r_fft = FFTRoutine::FFTW_ESTIMATE_FFT,
    const std::string& fft_wisdom_filename = std::string())
{
    assert(kernel_exact || (oversampling >= 1)); // If kernel exact is false, then oversampling must be >= 1
    assert(image_size > 0);
    assert(kernel_support > 0);
    assert(cell_size > 0.0);
    assert(vis.n_elem == vis_weights.n_elem);

#ifdef FUNCTION_TIMINGS
    times_iv.reserve(NUM_TIME_INST);
    times_iv.push_back(std::chrono::high_resolution_clock::now());
#endif

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double grid_pixel_width_lambda = (1.0 / (arc_sec_to_rad(cell_size) * double(image_size)));
    arma::mat uv_in_pixels = (uvw_lambda / grid_pixel_width_lambda);
    arma::cx_mat conv_vis = vis;
    arma::mat conv_vis_weights = vis_weights;

    // Remove W column
    uv_in_pixels.shed_col(2);

    // If a visibility point is located in the top half-plane, move it to the bottom half-plane to a symmetric position with respect to the matrix centre (0,0)
    convert_to_halfplane_visibilities(uv_in_pixels, conv_vis, conv_vis_weights, kernel_support);

    // Perform convolutional gridding of complex visibilities
    GridderOutput gridded_data;
    if (generate_beam) {
        gridded_data = convolve_to_grid<true>(kernel_creator, kernel_support, image_size, uv_in_pixels, conv_vis, conv_vis_weights, kernel_exact, oversampling);
    } else {
        gridded_data = convolve_to_grid<false>(kernel_creator, kernel_support, image_size, uv_in_pixels, conv_vis, conv_vis_weights, kernel_exact, oversampling);
    }

#ifdef FUNCTION_TIMINGS
    times_iv.push_back(std::chrono::high_resolution_clock::now());
#endif

    arma::Mat<real_t> fft_result_image;
    arma::Mat<real_t> fft_result_beam;

    // Reuse gridded_data buffer if FFT is INPLACE
    if (r_fft == stp::FFTRoutine::FFTW_WISDOM_INPLACE_FFT) {
        fft_result_image = std::move(arma::Mat<real_t>(reinterpret_cast<real_t*>(gridded_data.vis_grid.memptr()), (gridded_data.vis_grid.n_rows) * 2, gridded_data.vis_grid.n_cols, false, false));
        fft_result_beam = std::move(arma::Mat<real_t>(reinterpret_cast<real_t*>(gridded_data.sampling_grid.memptr()), (gridded_data.sampling_grid.n_rows) * 2, gridded_data.sampling_grid.n_cols, false, false));
    }

    // Run iFFT over convolved matrices
    // First: FFT of image matrix
    fft_fftw_c2r(gridded_data.vis_grid, fft_result_image, r_fft, fft_wisdom_filename);
    // Delete gridded image matrix (only if FFT is not inplace)
    if (r_fft != stp::FFTRoutine::FFTW_WISDOM_INPLACE_FFT) {
        gridded_data.vis_grid.delete_matrix_buffer();
    }
#ifdef FFTSHIFT
    fftshift(fft_result_image);
#endif

    // Second: FFT of beam matrix (optional)
    if (generate_beam) {
        fft_fftw_c2r(gridded_data.sampling_grid, fft_result_beam, r_fft, fft_wisdom_filename);
        // Delete gridded beam matrix (only if FFT is not inplace)
        if (r_fft != stp::FFTRoutine::FFTW_WISDOM_INPLACE_FFT) {
            gridded_data.sampling_grid.delete_matrix_buffer();
        }
#ifdef FFTSHIFT
        fftshift(fft_result_beam);
#endif
    }

#ifdef FUNCTION_TIMINGS
    times_iv.push_back(std::chrono::high_resolution_clock::now());
#endif

    // Normalize image and beam
    if (gridded_data.sample_grid_total > 0.0) {
        real_t normalization_factor = 1.0 / (gridded_data.sample_grid_total);
        // Image
        tbb::parallel_for(tbb::blocked_range<size_t>(0, fft_result_image.n_elem), [&](const tbb::blocked_range<size_t>& r) {
            for (arma::uword i = r.begin(); i < r.end(); ++i) {
                fft_result_image[i] *= normalization_factor;
            }
        });
        // Beam is optional
        if (generate_beam) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, fft_result_beam.n_elem), [&](const tbb::blocked_range<size_t>& r) {
                for (arma::uword i = r.begin(); i < r.end(); ++i) {
                    fft_result_beam[i] *= normalization_factor;
                }
            });
        }
    }

#ifdef FUNCTION_TIMINGS
    times_iv.push_back(std::chrono::high_resolution_clock::now());
#endif

    return std::make_pair(std::move(fft_result_image), std::move(fft_result_beam));
}
}
#endif /* IMAGER_H */
