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
     * @param[in] kernel_size (int): W-Kernel size (work area size)
     * @param[in] cell_size (double): Cell size
     * @param[in] oversampling (int): Kernel oversampling
     * @param[in] w_proj (W_ProjectionPars): W-Projection parameters
     * @param[in] r_fft (FFTRoutine): Selects FFT routine.
     */
    WideFieldImaging(size_t kernel_size, double cell_size, int oversampling, int undersampling, const W_ProjectionPars& w_proj,
        FFTRoutine r_fft = FFTRoutine::FFTW_ESTIMATE_FFT);

    /**
     * @brief Generate combined W-AA kernel (does not perform Fourier Transform).
     *
     * This function is useful for A-projection as the resulting kernel is then combined with A-kernel multiple times (for each timestep).
     *
     * @param[in] input_w_value (double): Average W value used to compute W-kernel.
     * @param[in] aa_kernel_img (arma::Col): Sampled image-domain anti-aliasing kernel.
     */
    void generate_combined_w_aa_kernel(double input_w_value, const arma::Col<real_t>& aa_kernel_img);

    /**
     * @brief Generate convolution kernel at oversampled-pixel offsets for A-Projection.
     *
     * @param[in] a_kernel_img (arma::Mat): A-kernel (computed from beam pattern) used to generate A-projection convolution kernel.
     */
    void generate_convolution_kernel_aproj(const arma::Mat<real_t>& a_kernel_img);

    /**
    * @brief Generate convolution kernel at oversampled-pixel offsets for W-Projection.
    *
    * @param[in] input_w_value (double): Average W value used to compute W-kernel.
    * @param[in] aa_kernel_img (arma::Col<real_t>&): Sampled image-domain anti-aliasing kernel.
    */
    void generate_convolution_kernel_wproj(double input_w_value, const arma::Col<real_t>& aa_kernel_img);

    /**
     * @brief Generate a cache of kernels at oversampled-pixel offsets for A/W-Projection.
     *
     * @return (arma::field<arma::mat>): Cache of convolution kernels associated to oversampling-pixel offsets.
     */
    arma::field<arma::Mat<cx_real_t>> generate_kernel_cache();

    /**
     * @brief Get kernel truncation value in pixels.
     *
     * @return (int): Kernel truncation value.
     */
    int get_trunc_conv_support()
    {
        return truncated_wpconv_support;
    }

private:
    /**
     * @brief Combine W-kernel and anti-aliasing kernel.
     *
     * W-kernel is internally generated using the input cell size.
     *
     * @param[in] aa_kernel_img (arma::Col): Sampled image-domain anti-aliasing kernel.
     * @param[in] scaled_cell_size (double): Scaled cell size for W-kernel generation.
     */
    MatStp<cx_real_t> combine_kernels(const arma::Col<real_t>& aa_kernel_img, const double scaled_cell_size);

    /**
     * @brief Auxiliary function to combine w_kernel and aa_img_domain_kernel values
     */
    inline cx_real_t combine_2d_kernel_value(const arma::Col<real_t>& aa_kernel_img,
        const int pixel_x, const int pixel_y, const real_t fft_factor,
        const double w_value, const double scaled_cell_size, const int ctr_idx);

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
     * @param[in] trunc_at (double): Truncation percentage.
     * @return (arma::Col): Interpolation values at kernel half quadrant
     */
    template <bool isCubic>
    arma::Col<cx_real_t> RadialInterpolate(const arma::Col<real_t>& x_array, const arma::Col<cx_real_t>& y_array, double trunc_at)
    {
        assert(x_array.n_elem <= y_array.n_elem);
        tk::spline<isCubic> m_spline_real;
        tk::spline<isCubic> m_spline_imag;

        if (trunc_at > 0.0) {
            //truncate kernel at a certain percentage from maximum
            double scale = M_SQRT2 / double(oversampling * 2);
            real_t min_value = std::abs(y_array[0]) * trunc_at;
            int trunc_idx = wp.max_wpconv_support * oversampling;
            while (trunc_idx > oversampling) {
                if (std::abs(y_array[trunc_idx]) > min_value) {
                    break;
                }
                trunc_idx -= oversampling;
            }
            truncated_wpconv_support = std::min(int((ceil(double(trunc_idx) * scale))), wp.max_wpconv_support);
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
            k = size_t(new_x * M_SQRT2);
            // numeric imprecision due to the M_SQRT1_2 division don't allow this next step to be skipped
            // the speed gain will still be enourmous in comparison with the previous algorithm (that searched the whole array)
            if (new_x < x_array(k)) {
                if (k > 0)
                    k--;
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
        return std::move(tmp_kernel_half_quandrant);
    }

    int max_hankel_kernel_size;
    int current_hankel_kernel_size;
    int truncated_wpconv_support;
    double w_value;
    size_t array_size;

    size_t kernel_size;
    double cell_size;
    int oversampling;
    int undersampling;
    W_ProjectionPars wp;
    FFTRoutine r_fft;

    MatStp<cx_real_t> comb_kernel;
    arma::Col<cx_real_t> kernel_half_quandrant;
    arma::Mat<cx_real_t> conv_kernel;
    arma::Mat<real_t> DHT;
    arma::Col<real_t> hankel_radius_points;
};

/**
 * @brief Compute parallatic angle for beam rotation.
 *
 *  @param[in] ha (double): Hour angle of the object, in decimal hours (0,24)
 *  @param[in] dec_d (double): Declination of the object, in degrees
 *  @param[in] lat_d (double): The latitude of the observer, in degrees
 *  @return (double): Parallactic angle in radians
 */
double parangle(double ha, double dec_d, double lat_d);
}
#endif /* AW_PROJECTION_H */
