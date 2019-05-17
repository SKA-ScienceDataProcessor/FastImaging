/** @file aw_projection.h
 *  @brief Classes and function prototypes of aw_projection.
 */

#ifndef AW_PROJECTION_H
#define AW_PROJECTION_H

#include "../common/fft.h"
#include "../common/matstp.h"
#include "../common/spline.h"
#include "../types.h"
#include <armadillo>

namespace stp {

/**
 * @brief The WideFieldImaging class
 *
 * Implements most functions related to W-Projection and A-Projection
 */
class WideFieldImaging {
public:
    /**
     * @brief Default constructor
     */
    WideFieldImaging() = default;

    /**
     * @brief WideFieldImaging constructor initilization
     * @param[in] kernel_size (uint): W-Kernel size (work area size)
     * @param[in] cell_size (double): Cell size
     * @param[in] oversampling (uint): Kernel oversampling ratio
     * @param[in] scaling_factor (double): Scaling factor
     * @param[in] w_proj (W_ProjectionPars): W-Projection parameters
     * @param[in] r_fft (FFTRoutine): Selects FFT routine.
     */
    WideFieldImaging(uint kernel_size, double cell_size, uint oversampling, double scaling_factor, const W_ProjectionPars& w_proj,
        FFTRoutine r_fft = FFTRoutine::FFTW_ESTIMATE_FFT);

    /**
     * @brief Generate image-domain convolution kernel by multiplying image-domain anti-aliasing and W-kernels.
     *
     * Generating the image-domain convolution kernel in a separate function is useful for A-projection, because this kernel needs to be multiplied
     * by A-kernel several times (for each timestep).
     *
     * @param[in] aa_kernel_img (arma::Col): Sampled image-domain anti-aliasing kernel.
     * @param[in] input_w_value (real_t): Average W value used to compute W-kernel.
     */
    void generate_image_domain_convolution_kernel(const real_t input_w_value, const arma::Col<real_t>& aa_kernel_img);

    /**
     * @brief Generate projected image-domain convolution kernel obtained by multiplying image-domain anti-aliasing and W-kernels.
     *
     * The projected image-domain convolution kernel is used for Hankel transform using the projection slice theorem.
     *
     * @param[in] aa_kernel_img (arma::Col): Sampled image-domain anti-aliasing kernel.
     * @param[in] input_w_value (real_t): Average W value used to compute W-kernel.
     */
    arma::Col<cx_real_t> generate_projected_image_domain_kernel(const real_t input_w_value, const arma::Col<real_t>& aa_kernel_img);

    /**
    * @brief Generate convolution kernel at oversampled-pixel offsets for W-Projection.
    *
    * @param[in] input_w_value (real_t): Average W value used to compute W-kernel.
    * @param[in] aa_kernel_img (arma::Col<real_t>&): Sampled image-domain anti-aliasing kernel.
    * @param[in] hankel_proj_slice (bool): Whether to use projection slice theorem for Hankel transform or not
    */
    void generate_convolution_kernel_wproj(real_t input_w_value, const arma::Col<real_t>& aa_kernel_img, bool hankel_proj_slice = false);

    /**
     * @brief Generate convolution kernel at oversampled-pixel offsets for A-Projection.
     *
     * @param[in] a_kernel_img (arma::Mat): A-kernel (computed from beam pattern) used to generate A-projection convolution kernel.
     */
    void generate_convolution_kernel_aproj(const arma::Mat<real_t>& a_kernel_img);

    /**
     * @brief Generate a cache of kernels at oversampled-pixel offsets for A/W-Projection.
     *
     * @return (arma::field<arma::mat>): Cache of convolution kernels associated to oversampling-pixel offsets.
     */
    arma::field<arma::Mat<cx_real_t>> generate_kernel_cache();

    /**
     * @brief Get kernel truncation value in pixels.
     *
     * @return (uint): Kernel truncation value.
     */
    uint get_trunc_conv_support() const
    {
        return truncated_wpconv_support;
    }

    // Upsampled convolution kernel
    arma::Mat<cx_real_t> conv_kernel;

private:
    /**
     * @brief Auxiliary function to combine w_kernel and aa_img_domain_kernel values
     */
    inline cx_real_t combine_2d_kernel_value(const arma::Col<real_t>& aa_kernel_img,
        const size_t pixel_x, const size_t pixel_y, const real_t w_value, const real_t scaled_cell_size, const uint ctr_idx);

    /**
     * @brief Generate kernel diagonal radius points for interpolation.
     *
     * @param[in] arrsize (size_t): Kernel size.
     */
    arma::Col<real_t> generate_hankel_radius_points(size_t arrsize);

    /**
     * @brief Compute Hankel transformation matrix.
     *
     * @param[in] arrsize (size_t): Hankel transform size.
     */
    arma::Mat<real_t> dht(size_t arrsize); // Generate DHT Matrix

    /**
     * @brief Perform radial kernel interpolation in kernel half quadrant given the radius function values and radius points.
     *
     * @param[in] x_array (arma::Col): Radius points.
     * @param[in] y_array (arma::Col): Radius function values.
     * @param[in] trunc_at (real_t): Truncation percentage.
     * @return (arma::Col): Interpolation values at kernel half quadrant
     */
    template <bool isCubic, bool isDiagonal = false>
    arma::Col<cx_real_t> RadialInterpolate(const arma::Col<real_t>& x_array, const arma::Col<cx_real_t>& y_array, real_t trunc_at)
    {
        assert(x_array.n_elem <= y_array.n_elem);
        tk::spline<isCubic> m_spline_real;
        tk::spline<isCubic> m_spline_imag;

        if (trunc_at > real_t(0.0)) {
            //truncate kernel at a certain percentage from maximum
            real_t min_value = std::abs(y_array[0]) * trunc_at;
            uint trunc_idx = wp.max_wpconv_support * oversampling;
            real_t scale = real_t(1.0) / real_t(oversampling);
            if (isDiagonal == true) {
                scale = real_t(M_SQRT2) / real_t(oversampling * 2);
            }
            while (trunc_idx > oversampling) {
                if (std::abs(y_array[trunc_idx]) > min_value) {
                    break;
                }
                trunc_idx -= oversampling;
            }
            truncated_wpconv_support = std::min(uint((std::ceil(real_t(trunc_idx) * scale))), wp.max_wpconv_support);
        }

        current_hankel_kernel_size = (truncated_wpconv_support * 2 + 1 + 1) * oversampling;

        const size_t kernel_offset = current_hankel_kernel_size / 2;
        const size_t final_kernel_size = kernel_offset + (kernel_offset * (kernel_offset - 1) / 2);
        arma::Col<cx_real_t> tmp_kernel_half_quandrant(final_kernel_size);

        // Extract real and imaginary parts
        size_t n_elems = y_array.n_elem;
        arma::Col<real_t> y_array_real(n_elems);
        arma::Col<real_t> y_array_imag(n_elems);
        for (size_t i = 0; i < n_elems; i++) {
            y_array_real(i) = reinterpret_cast<const real_t(&)[2]>(y_array(i))[0];
            y_array_imag(i) = reinterpret_cast<const real_t(&)[2]>(y_array(i))[1];
        }

        m_spline_real.set_points(x_array, y_array_real, false);
        m_spline_imag.set_points(x_array, y_array_imag, false);

        real_t new_x;
        size_t k;
        size_t ii = 0, j = 0;
        for (size_t i = 0; i < final_kernel_size; i++, j++) {
            if (j == kernel_offset) {
                j = ++ii;
            }
            if (j == 0)
                new_x = ii;
            else if (ii == 0)
                new_x = j;
            else
                new_x = std::sqrt(static_cast<real_t>(ii * ii + j * j));

            // find where the interpolation is going to happen
            if (isDiagonal == true) {
                k = size_t(new_x * real_t(M_SQRT2));
                // numeric imprecision due to the M_SQRT1_2 division don't allow this next step to be skipped
                // the speed gain will still be enourmous in comparison with the previous algorithm (that searched the whole array)
                if (new_x < x_array(k)) {
                    if (k > 0)
                        k--;
                }
            } else {
                k = size_t(new_x);
                if (k >= x_array.n_elem)
                    k = x_array.n_elem - 1;
            }
            /* kernel_half_quandrant is an array that stores all the interpolated values of half a quadrant of the final kernel
             * in the following index order:
             *     .......
             *     .......
             *     .......
             *     ...0123
             *     ....456
             *     .....78
             *     ......9
             */

            real_t val_re = m_spline_real(new_x, k, x_array[k], y_array_real[k]);
            real_t val_im = m_spline_imag(new_x, k, x_array[k], y_array_imag[k]);
            assert(arma::is_finite(val_re));
            assert(arma::is_finite(val_im));
            tmp_kernel_half_quandrant[i] = cx_real_t(val_re, val_im);
        }
        return tmp_kernel_half_quandrant;
    }

    uint max_hankel_kernel_size;
    uint current_hankel_kernel_size;
    uint truncated_wpconv_support;
    real_t w_value;
    uint array_size;

    uint kernel_size;
    real_t cell_size;
    uint oversampling;
    double scaling_factor;
    W_ProjectionPars wp;
    FFTRoutine r_fft;

    MatStp<cx_real_t> comb_kernel;
    arma::Col<cx_real_t> kernel_half_quandrant;
    arma::Mat<real_t> DHT;
    arma::Col<real_t> hankel_radius_points;
};

/**
 * @brief Compute parallatic angle for beam rotation.
 *
 *  @param[in] ha (real_t): Hour angle of the object, in decimal hours (0,24)
 *  @param[in] dec_rad (real_t): Declination of the object, in radians
 *  @param[in] ra_rad (real_t): Right Ascension of the object, in radians
 *  @return (real_t): Parallactic angle in radians
 */
real_t parangle(real_t ha, real_t dec_rad, real_t ra_rad);
}
#endif /* AW_PROJECTION_H */
