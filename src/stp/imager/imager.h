/**
 * @file imager.h
 * @brief Function prototypes of the imager.
 */

#ifndef IMAGER_H
#define IMAGER_H

// STP library includes
#include "../common/fft.h"
#include "../global_macros.h"
#include "../gridder/gridder.h"
#include "../types.h"
#include <fftw3.h>
#include <thread>

namespace stp {

/**
 * @brief Normalizes the result image and beam.
 *
 * @param[in] image_mat (std::pair<arma::mat): Image matrix.
 * @param[in] beam_mat (std::pair<arma::mat): Beam model matrix.
 * @param[in] kernel_creator (typename T): Callable object that returns a convolution kernel.
 * @param[in] image_size (int): Width of the image in pixels.
 * @param[in] normalization_factor (int): Normalization factor computed from sampling grid.
 * @param[in] analytic_gcf (bool): Compute approximation of image-domain kernel from analytic expression or DFT. Default is true.
 * @param[in] generate_beam (bool): Enables generation of gridded sampling matrix. Default is false.
 * @param[in] r_fft (FFTRoutine): Selects FFT routine.
 */
template <bool grid_correction, typename T>
void normalise_image_beam_result_1D(
    arma::Mat<real_t>& image_mat,
    arma::Mat<real_t>& beam_mat,
    arma::Mat<real_t>& norm_image,
    arma::Mat<real_t>& norm_beam,
    const T& kernel_creator,
    const size_t padded_image_size,
    const size_t image_size,
    const real_t normalization_factor,
    const bool analytic_gcf = true,
    const bool generate_beam = false,
    FFTRoutine r_fft = FFTRoutine::FFTW_ESTIMATE_FFT)
{
    size_t half_padded_image_size = padded_image_size / 2;

    // generate ImgDomKernel
    arma::Col<real_t> fft_1D_array;
    if (grid_correction) {
        fft_1D_array = ImgDomKernel(kernel_creator, padded_image_size, false, analytic_gcf, r_fft);
    }

    // normalisation
#ifdef FFTSHIFT
    norm_image.set_size(image_size, image_size);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, image_size),
        [&](const tbb::blocked_range<size_t>& r) {
            size_t j_begin = r.begin(),
                   j_end = r.end();
            size_t orig_i = (half_padded_image_size - image_size / 2);
            size_t orig_j = (half_padded_image_size - image_size / 2);
            for (size_t j = j_begin; j < j_end; ++j, ++orig_j) {
                for (size_t i = 0; i < image_size; ++i, ++orig_i) {
                    if (grid_correction) {
                        norm_image.at(i, j) = image_mat(orig_i, orig_j) * normalization_factor / (fft_1D_array.at((half_padded_image_size - image_size / 2) + i) * fft_1D_array.at((half_padded_image_size - image_size / 2) + j));
                    } else {
                        norm_image.at(i, j) = image_mat(orig_i, orig_j) * normalization_factor;
                    }
                }
            }
        });

    image_mat.reset();

    // Beam is optional
    if (generate_beam) {
        norm_beam.set_size(image_size, image_size);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, image_size),
            [&](const tbb::blocked_range<size_t>& r) {
                size_t j_begin = r.begin(),
                       j_end = r.end();
                size_t orig_i = (half_padded_image_size - image_size / 2);
                size_t orig_j = (half_padded_image_size - image_size / 2);
                for (size_t j = j_begin; j < j_end; ++j, ++orig_j) {
                    for (size_t i = 0; i < image_size; ++i, ++orig_i) {
                        if (grid_correction) {
                            norm_beam.at(i, j) = beam_mat(orig_i, orig_j) * normalization_factor / (fft_1D_array.at((half_padded_image_size - image_size / 2) + i) * fft_1D_array.at((half_padded_image_size - image_size / 2) + j));
                        } else {
                            norm_image.at(i, j) = image_mat(orig_i, orig_j) * normalization_factor;
                        }
                    }
                }
            });
        beam_mat.reset();
    }
#else
    norm_image.set_size(image_size, image_size);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, image_size / 2),
        [&](const tbb::blocked_range<size_t>& r) {
            size_t j_begin = r.begin(),
                   j_end = r.end();
            for (size_t j = j_begin; j < j_end; ++j) {
                size_t jj = j + half_padded_image_size;
                size_t ii = half_padded_image_size;
                for (size_t i = 0; i < (image_size / 2); ++i, ++ii) {
                    if (grid_correction) {
                        norm_image.at(i, j) = image_mat(i, j) * normalization_factor / (fft_1D_array.at(ii) * fft_1D_array.at(jj));
                    } else {
                        norm_image.at(i, j) = image_mat(i, j) * normalization_factor;
                    }
                }
                ii = half_padded_image_size - image_size / 2;
                size_t orig_i = padded_image_size - (image_size / 2);
                for (size_t i = (image_size / 2); i < image_size; ++i, ++ii, ++orig_i) {
                    if (grid_correction) {
                        norm_image.at(i, j) = image_mat(orig_i, j) * normalization_factor / (fft_1D_array.at(ii) * fft_1D_array.at(jj));
                    } else {
                        norm_image.at(i, j) = image_mat(orig_i, j) * normalization_factor;
                    }
                }
            }
        });

    tbb::parallel_for(tbb::blocked_range<size_t>(image_size / 2, image_size),
        [&](const tbb::blocked_range<size_t>& r) {
            size_t j_begin = r.begin(),
                   j_end = r.end();
            size_t orig_j = padded_image_size + j_begin - image_size;
            for (size_t j = j_begin; j < j_end; ++j, ++orig_j) {
                size_t jj = half_padded_image_size + j - image_size;
                size_t ii = half_padded_image_size;
                for (size_t i = 0; i < (image_size / 2); ++i, ++ii) {
                    if (grid_correction) {
                        norm_image.at(i, j) = image_mat(i, orig_j) * normalization_factor / (fft_1D_array.at(ii) * fft_1D_array.at(jj));
                    } else {
                        norm_image.at(i, j) = image_mat(i, orig_j) * normalization_factor;
                    }
                }
                ii = half_padded_image_size - image_size / 2;
                size_t orig_i = padded_image_size - (image_size / 2);
                for (size_t i = (image_size / 2); i < image_size; ++i, ++ii, ++orig_i) {
                    if (grid_correction) {
                        norm_image.at(i, j) = image_mat(orig_i, orig_j) * normalization_factor / (fft_1D_array.at(ii) * fft_1D_array.at(jj));
                    } else {
                        norm_image.at(i, j) = image_mat(orig_i, orig_j) * normalization_factor;
                    }
                }
            }
        });

    image_mat.reset();

    // Beam is optional
    if (generate_beam) {
        norm_beam.set_size(image_size, image_size);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, image_size / 2),
            [&](const tbb::blocked_range<size_t>& r) {
                size_t j_begin = r.begin(),
                       j_end = r.end();
                for (size_t j = j_begin; j < j_end; ++j) {
                    size_t jj = j + half_padded_image_size;
                    size_t ii = half_padded_image_size;
                    for (size_t i = 0; i < (image_size / 2); ++i, ++ii) {
                        if (grid_correction) {
                            norm_beam.at(i, j) = beam_mat(i, j) * normalization_factor / (fft_1D_array.at(ii) * fft_1D_array.at(jj));
                        } else {
                            norm_beam.at(i, j) = beam_mat(i, j) * normalization_factor;
                        }
                    }
                    ii = half_padded_image_size - image_size / 2;
                    size_t orig_i = padded_image_size - (image_size / 2);
                    for (size_t i = (image_size / 2); i < image_size; ++i, ++ii, ++orig_i) {
                        if (grid_correction) {
                            norm_beam.at(i, j) = beam_mat(orig_i, j) * normalization_factor / (fft_1D_array.at(ii) * fft_1D_array.at(jj));
                        } else {
                            norm_beam.at(i, j) = beam_mat(orig_i, j) * normalization_factor;
                        }
                    }
                }
            });

        tbb::parallel_for(tbb::blocked_range<size_t>(image_size / 2, image_size),
            [&](const tbb::blocked_range<size_t>& r) {
                size_t j_begin = r.begin(),
                       j_end = r.end();
                size_t orig_j = padded_image_size + j_begin - image_size;
                for (size_t j = j_begin; j < j_end; ++j, ++orig_j) {
                    size_t jj = half_padded_image_size + j - image_size;
                    size_t ii = half_padded_image_size;
                    for (size_t i = 0; i < (image_size / 2); ++i, ++ii) {
                        if (grid_correction) {
                            norm_beam.at(i, j) = beam_mat(i, orig_j) * normalization_factor / (fft_1D_array.at(ii) * fft_1D_array.at(jj));
                        } else {
                            norm_beam.at(i, j) = beam_mat(i, orig_j) * normalization_factor;
                        }
                    }
                    ii = half_padded_image_size - image_size / 2;
                    size_t orig_i = padded_image_size - (image_size / 2);
                    for (size_t i = (image_size / 2); i < image_size; ++i, ++ii, ++orig_i) {
                        if (grid_correction) {
                            norm_beam.at(i, j) = beam_mat(orig_i, orig_j) * normalization_factor / (fft_1D_array.at(ii) * fft_1D_array.at(jj));
                        } else {
                            norm_beam.at(i, j) = beam_mat(orig_i, orig_j) * normalization_factor;
                        }
                    }
                }
            });

        beam_mat.reset();
    }
#endif
}

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
 * @param[in] img_pars (ImagerPars): Imager parameters (see ImagerPars struct).
 * @param[in] w_proj (W_ProjectionPars): W-projection parameters (see W_ProjectionPars struct).
 * @param[in] a_proj (A_ProjectionPars): A-projection parameters (see A_ProjectionPars struct).
 *
 * @return (std::pair<arma::mat, arma::mat>): Two matrices representing the generated image map and beam model (image, beam).
 */
template <typename T>
std::pair<arma::Mat<real_t>, arma::Mat<real_t>> image_visibilities(
    const T kernel_creator,
    const arma::cx_mat& vis,
    const arma::mat& vis_weights,
    const arma::mat& uvw_lambda,
    const ImagerPars& img_pars = ImagerPars(),
    const W_ProjectionPars& w_proj = W_ProjectionPars(),
    const A_ProjectionPars& a_proj = A_ProjectionPars())
{
#ifdef FUNCTION_TIMINGS
    times_iv.reserve(NUM_TIME_INST);
    times_gridder.reserve(NUM_TIME_INST);
#endif
    TIMESTAMP_IMAGER

    int padded_image_size = img_pars.padded_image_size;
    double cell_size = img_pars.cell_size;
    bool kernel_exact = img_pars.kernel_exact;
    int kernel_support = img_pars.kernel_support;
    int oversampling = img_pars.oversampling;
    bool generate_beam = img_pars.generate_beam;
    FFTRoutine r_fft = img_pars.r_fft;

    /* Some checks */
    assert(img_pars.padding_factor >= 1.0);
    assert(padded_image_size >= img_pars.image_size);
    assert(kernel_exact || (oversampling >= 1)); // If kernel exact is false, then oversampling must be >= 1
    assert(padded_image_size > 0);
    assert(ispowerof2(padded_image_size)); // Image size must be power of two. Also parallel complex2real FFTW function only works with image sizes multiple of 4.
    assert(kernel_support > 0);
    assert(cell_size > 0.0);
    assert(vis.n_elem == vis_weights.n_elem);
#ifdef WPROJECTION
    if (w_proj.num_wplanes > 0) {
        assert(w_proj.isEnabled());
    }
#else
    assert(!w_proj.isEnabled());
#endif
#ifdef APROJECTION
    // A-proj can be used only with W-proj enabled
    if (a_proj.num_timesteps > 0) {
        assert(a_proj.isEnabled());
        assert(w_proj.isEnabled());
        assert(!w_proj.hankel_opt);
        if (!w_proj.isEnabled())
            throw std::runtime_error("W-projection must be used when A-projection is enabled.");
        if (w_proj.hankel_opt)
            throw std::runtime_error("Hankel transform cannot be used when A-projection is enabled.");
    }
#else
    assert(!a_proj.isEnabled());
#endif
    assert(!(kernel_exact && (w_proj.isEnabled()))); // W-proj cannot be used when 'kernel_exact' is true.

    // Init FFTW threads
    init_fftw(r_fft, img_pars.fft_wisdom_filename);

    // Size of a UV-grid pixel, in multiples of wavelength (lambda):
    double inv_grid_pixel_width_lambda = arc_sec_to_rad(cell_size) * double(padded_image_size);
    // convert u,v to pixel
    arma::mat uv_lambda(uvw_lambda.n_rows, 2);
    arma::vec w_lambda = uvw_lambda.col(2);
    for (size_t idx = 0; idx < uvw_lambda.n_rows; ++idx) {
        uv_lambda.at(idx, 0) = uvw_lambda.at(idx, 0) * inv_grid_pixel_width_lambda;
        uv_lambda.at(idx, 1) = uvw_lambda.at(idx, 1) * inv_grid_pixel_width_lambda;
    }

    // Perform convolutional gridding of complex visibilities
    GridderOutput gridded_data;
    bool shift_uv = true;
    bool halfplane_gridding = true;

    if (generate_beam) {
        gridded_data = convolve_to_grid<true>(kernel_creator, kernel_support, padded_image_size,
            uv_lambda, vis, vis_weights, kernel_exact, oversampling, shift_uv, halfplane_gridding,
            w_proj, w_lambda, cell_size, img_pars.analytic_gcf, r_fft, a_proj);
    } else {
        gridded_data = convolve_to_grid<false>(kernel_creator, kernel_support, padded_image_size,
            uv_lambda, vis, vis_weights, kernel_exact, oversampling, shift_uv, halfplane_gridding,
            w_proj, w_lambda, cell_size, img_pars.analytic_gcf, r_fft, a_proj);
    }

    TIMESTAMP_IMAGER

    arma::Mat<real_t> fft_result_image;
    arma::Mat<real_t> fft_result_beam;

    // Reuse gridded_data buffer if FFT is INPLACE
    if (r_fft == stp::FFTRoutine::FFTW_WISDOM_INPLACE_FFT) {
        fft_result_image = std::move(arma::Mat<real_t>(reinterpret_cast<real_t*>(gridded_data.vis_grid.memptr()), (gridded_data.vis_grid.n_rows) * 2, gridded_data.vis_grid.n_cols, false, false));
        fft_result_beam = std::move(arma::Mat<real_t>(reinterpret_cast<real_t*>(gridded_data.sampling_grid.memptr()), (gridded_data.sampling_grid.n_rows) * 2, gridded_data.sampling_grid.n_cols, false, false));
    }

    // Run iFFT over convolved matrices
    // First: FFT of image matrix
    fft_fftw_c2r(gridded_data.vis_grid, fft_result_image, r_fft);
    // Delete gridded image matrix (only if FFT is not inplace)
    if (r_fft != stp::FFTRoutine::FFTW_WISDOM_INPLACE_FFT) {
        gridded_data.vis_grid.reset();
    }
#ifdef FFTSHIFT
    fftshift(fft_result_image);
#endif

    // Second: FFT of beam matrix (optional)
    if (generate_beam) {
        fft_fftw_c2r(gridded_data.sampling_grid, fft_result_beam, r_fft);
        // Delete gridded beam matrix (only if FFT is not inplace)
        if (r_fft != stp::FFTRoutine::FFTW_WISDOM_INPLACE_FFT) {
            gridded_data.sampling_grid.reset();
        }
#ifdef FFTSHIFT
        fftshift(fft_result_beam);
#endif
    }
    TIMESTAMP_IMAGER

    // Normalisation and convolution kernel correction
    arma::Mat<real_t> norm_result_image;
    arma::Mat<real_t> norm_result_beam;
    if (gridded_data.sample_grid_total > 0.0) {
        real_t normalization_factor = 1.0 / (gridded_data.sample_grid_total);
        if (img_pars.gridding_correction == true) {
            normalise_image_beam_result_1D<true>(fft_result_image, fft_result_beam, norm_result_image, norm_result_beam, kernel_creator, padded_image_size,
                img_pars.image_size, normalization_factor, img_pars.analytic_gcf, generate_beam, r_fft);
        } else {
            normalise_image_beam_result_1D<false>(fft_result_image, fft_result_beam, norm_result_image, norm_result_beam, kernel_creator, padded_image_size,
                img_pars.image_size, normalization_factor, img_pars.analytic_gcf, generate_beam, r_fft);
        }
    }

    TIMESTAMP_IMAGER

    // Destroy FFTW threads
#ifdef USE_FLOAT
    fftwf_cleanup_threads();
#else
    fftw_cleanup_threads();
#endif

    return std::make_pair(std::move(norm_result_image), std::move(norm_result_beam));
}

/**
 * @brief ImageVisibilities class. Runs imager.
 */
class ImageVisibilities {
public:
    /**
     * @brief Delete default constructor
     */
    ImageVisibilities() = delete;

    /**
     * @brief ImageVisibilities constructor
     *
     * Calls image_visbilities function
     *
     * @param[in] vis (arma::cx_mat): Complex visibilities (1D array).
     * @param[in] vis_weights (arma::mat): Visibility weights (1D array).
     * @param[in] uvw_lambda (arma::mat): UVW-coordinates of complex visibilities. Units are multiples of wavelength.
     *                                    2D double array with 3 columns. Assumed ordering is u,v,w.
     * @param[in] img_pars (ImagerPars): Imager parameters (see ImagerPars struct).
     * @param[in] w_proj (W_ProjectionPars): W-projection parameters (see W_ProjectionPars struct).
     * @param[in] a_proj (A_ProjectionPars): A-projection parameters (see A_ProjectionPars struct).
     */
    ImageVisibilities(const arma::cx_mat& vis,
        const arma::mat& vis_weights,
        const arma::mat& uvw_lambda,
        const ImagerPars& img_pars,
        const W_ProjectionPars& w_proj,
        const A_ProjectionPars& a_proj);

    // Store gridded image and beam
    arma::Mat<real_t> vis_grid;
    arma::Mat<real_t> sampling_grid;
};
}
#endif /* IMAGER_H */
