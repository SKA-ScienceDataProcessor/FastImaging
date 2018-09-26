/**
 * @file aw_projection.cpp
 * @brief Implementation of the wplane functions.
 */

#include <cmath>
#include <tbb/tbb.h>

#include "../common/fft.h"
#include "../global_macros.h"
#include "aw_projection.h"

namespace stp {

// WideFieldImaging class constructor

WideFieldImaging::WideFieldImaging(size_t _kernel_size, double _cell_size, int _oversampling, int _undersampling,
    const W_ProjectionPars& _w_proj, FFTRoutine _r_fft)
    : w_value(0.0)
    , kernel_size(_kernel_size)
    , cell_size(_cell_size)
    , oversampling(_oversampling)
    , undersampling(_undersampling)
    , wp(_w_proj)
    , r_fft(_r_fft)
{
    array_size = kernel_size * oversampling;
    if (array_size % 2)
        throw std::runtime_error("array_size must be multiple of 2 for WideFieldImaging generation.");

    if (wp.hankel_opt == true) {
        // create auxiliary arrays/matrices
        max_hankel_kernel_size = (wp.max_wpconv_support * 2 + 1 + 1) * oversampling;
        DHT = dht(array_size / 2);
        hankel_radius_points = generate_hankel_radius_points(array_size / 2);
    }
    if (array_size < 4) {
        r_fft = FFTRoutine::FFTW_ESTIMATE_FFT;
    }

    truncated_wpconv_support = wp.max_wpconv_support;
    current_hankel_kernel_size = wp.max_wpconv_support;
}

// WideFieldImaging class Members

void WideFieldImaging::generate_combined_w_aa_kernel(double input_w_value, const arma::Col<real_t>& aa_kernel_img)
{
    // Set w_plane
    w_value = input_w_value;
    double scaled_cell_size = cell_size * undersampling;

    comb_kernel = combine_kernels(aa_kernel_img, scaled_cell_size);
}

void WideFieldImaging::generate_convolution_kernel_aproj(const arma::Mat<real_t>& a_kernel_img)
{
    assert(!comb_kernel.empty());
    assert(wp.hankel_opt == false);
    if (wp.hankel_opt == true) {
        throw std::runtime_error("A-Projection does not support Hankel transform optimization.");
    }

    double trunc_at = 0.0;
    if (wp.kernel_trunc_perc > 0.0)
        trunc_at = wp.kernel_trunc_perc / 100.0;

    assert(a_kernel_img.n_cols == kernel_size);

    // Multiply aa/w_kernel by a_kernel
    /* Note that non-null samples exist ony at these positions (x):
     *
     *      xxxx ... xxxx
     *      xxxx ... xxxx
     *      xxxx ... xxxx
     *      ...  ... ...
     *      xxxx ... xxxx
     *      xxxx ... xxxx
     *      xxxx ... xxxx
     */

    size_t half_kernel_size = kernel_size / 2;
    MatStp<cx_real_t> tmp_kernel(array_size, array_size);
    // Left side
    tbb::parallel_for(tbb::blocked_range<size_t>(0, half_kernel_size),
        [&](const tbb::blocked_range<size_t>& r) {
            size_t j_begin = r.begin(), j_end = r.end();
            size_t jj = half_kernel_size + j_begin;
            for (size_t j = j_begin; j < j_end; ++j, ++jj) {
                for (size_t i = 0, ii = half_kernel_size; i < half_kernel_size; ++i, ++ii) {
                    tmp_kernel.at(i, j) = comb_kernel.at(i, j) * a_kernel_img.at(ii, jj);
                }
                for (size_t i = array_size - half_kernel_size, ii = 0; i < array_size; ++i, ++ii) {
                    tmp_kernel.at(i, j) = comb_kernel.at(i, j) * a_kernel_img.at(ii, jj);
                }
            }
        });
    // Right side
    tbb::parallel_for(tbb::blocked_range<size_t>(array_size - half_kernel_size, array_size),
        [&](const tbb::blocked_range<size_t>& r) {
            size_t j_begin = r.begin(), j_end = r.end();
            size_t jj = r.begin() + half_kernel_size - array_size;
            for (size_t j = j_begin; j < j_end; ++j, ++jj) {
                for (size_t i = 0, ii = half_kernel_size; i < half_kernel_size; ++i, ++ii) {
                    tmp_kernel.at(i, j) = comb_kernel.at(i, j) * a_kernel_img.at(ii, jj);
                }
                for (size_t i = array_size - half_kernel_size, ii = 0; i < array_size; ++i, ++ii) {
                    tmp_kernel.at(i, j) = comb_kernel.at(i, j) * a_kernel_img.at(ii, jj);
                }
            }
        });

    STPLIB_DEBUG(spdlog::get("stplib"), "W-Proj: C2C FFT size = {}", tmp_kernel.n_rows);

    fft_fftw_c2c(tmp_kernel, conv_kernel, r_fft);

    // truncate kernel at a certain percentage from maximum
    if (trunc_at > 0.0) {
        real_t min_value = std::abs(conv_kernel.at(0, 0)) * trunc_at;
        int i = std::min(wp.max_wpconv_support * oversampling, int(array_size / 2));

        while (i > 0) {
            if (std::abs(conv_kernel.at(i, 0)) > min_value) {
                break;
            }
            i -= oversampling;
        }
        truncated_wpconv_support = std::min(int((i) / oversampling), wp.max_wpconv_support);
    }
}

void WideFieldImaging::generate_convolution_kernel_wproj(double input_w_value, const arma::Col<real_t>& aa_kernel_img)
{
    // Set w_plane
    w_value = input_w_value;

    double scaled_cell_size = cell_size * undersampling;
    double trunc_at = 0;
    if (wp.kernel_trunc_perc > 0.0)
        trunc_at = wp.kernel_trunc_perc / 100;

    if (wp.hankel_opt == false) {
        comb_kernel = combine_kernels(aa_kernel_img, scaled_cell_size);

        STPLIB_DEBUG(spdlog::get("stplib"), "W-Proj: C2C FFT size = {}", comb_kernel.n_rows);

        fft_fftw_c2c(comb_kernel, conv_kernel, r_fft);
        comb_kernel.reset();

        // truncate kernel at a certain percentage from maximum
        if (trunc_at > 0.0) {
            real_t min_value = std::abs(conv_kernel.at(0, 0)) * trunc_at;
            int i = std::min(wp.max_wpconv_support * oversampling, int(array_size >> 1));

            while (i > 0) {
                if (std::abs(conv_kernel.at(i, 0)) > min_value) {
                    break;
                }
                i -= oversampling;
            }
            truncated_wpconv_support = std::min(int((i) / oversampling), wp.max_wpconv_support);
        }
    } else {
        /*** Alternative method using Hankel transform ***/
        // Distance from each pixel in the diagonal
        // direction to the kernel-centre position:
        size_t half_arr_size = array_size / 2;
        scaled_cell_size *= M_SQRT2;
        size_t half_kernel_size = kernel_size / 2;

        // Vars
        arma::Col<cx_real_t> comb_kernel_img_radius = arma::zeros<arma::Col<cx_real_t>>(half_arr_size);

        /*********************************************/
        /*** generate combined kernel radius vector **/
        /*********************************************/
        for (size_t i = 0; i < half_kernel_size; ++i) {
            double distance_x = double(i) * scaled_cell_size;
            double squared_radians = distance_x * distance_x;
            cx_real_t tmp;
            if (squared_radians > 1.0) {
                tmp = 1.0;
            } else {
                real_t n = std::sqrt(1.0 - squared_radians);
                cx_real_t im(0, (-2.0 * M_PI * w_value * (n - 1.0)));
                tmp = std::exp(im) / n;
            }

            comb_kernel_img_radius(i) = tmp * aa_kernel_img(half_kernel_size + i) * aa_kernel_img(half_kernel_size + i);
        }

        /*********************************************/
        /*** generate Hankel transform output array **/
        /*********************************************/
        arma::Col<cx_real_t> comb_kernel_radius(half_arr_size);

        STPLIB_DEBUG(spdlog::get("stplib"), "W-Proj: DHT size = {}", half_arr_size);

        for (size_t i = 0; i < half_arr_size; i++) {
            real_t acc1 = 0;
            real_t acc2 = 0;
            real_t* comb_k = reinterpret_cast<real_t(&)[2]>(comb_kernel_img_radius(0));
            const size_t n_elem = comb_kernel_img_radius.n_elem;
            for (size_t j = 0; j < n_elem; j++) {
                // Only use double type within the cycle for successful auto-vectorization
                acc1 += DHT(j, i) * comb_k[j * 2];
                acc2 += DHT(j, i) * comb_k[j * 2 + 1];
            }
            comb_kernel_radius(i) = { acc1, acc2 };
        }

        /*** generate Hankel transform kernel ***/
        if (wp.interp_type == stp::InterpType::CUBIC)
            kernel_half_quandrant = RadialInterpolate<true>(hankel_radius_points, comb_kernel_radius, trunc_at);
        else
            kernel_half_quandrant = RadialInterpolate<false>(hankel_radius_points, comb_kernel_radius, trunc_at);
    }
}

arma::field<arma::Mat<cx_real_t>> WideFieldImaging::generate_kernel_cache()
{
    const size_t arr_size = array_size;
    const size_t ctr_idx = array_size / 2;
    const size_t oversamp = oversampling;
    const size_t max_conv_support = truncated_wpconv_support;
    const size_t oversampled_pixel = (oversamp / 2);
    const size_t cache_size = (oversampled_pixel * 2 + 1);
    const size_t tmp_cache_size = 2 * max_conv_support + 1;
    arma::field<arma::Mat<cx_real_t>> cache(cache_size, cache_size);
    arma::Mat<cx_real_t> tmp_cache(tmp_cache_size, tmp_cache_size);

    if (wp.hankel_opt == true) {
        //        const size_t kernel_offset = max_hankel_kernel_size / 2 + 1;
        const size_t kernel_offset = current_hankel_kernel_size / 2;
        const int oversamp_conv_ini = 0 - max_conv_support * oversamp + oversampled_pixel;
        const int oversamp_conv_end = 0 + max_conv_support * oversamp + oversampled_pixel;
        arma::Col<cx_real_t>* ptr = &kernel_half_quandrant;

        for (size_t x = 0; x < cache_size; x++) {
            for (size_t y = 0; y < cache_size; y++) {

                const int xx_start = oversamp_conv_ini - x;
                const int xx_end = oversamp_conv_end - x + 1;
                const int yy_start = oversamp_conv_ini - y;
                const int yy_end = oversamp_conv_end - y + 1;

                double kernel_sum = 0;
                size_t tx = 0, ty = 0, dcx = 0, dcy = 0, vec_idx = 0;
                for (int cx = xx_start; cx < xx_end; cx += oversamp) {
                    for (int cy = yy_start; cy < yy_end; cy += oversamp) {

                        dcx = cx > 0 ? cx : (-cx);
                        dcy = cy > 0 ? cy : (-cy);
                        if (dcx > dcy) {
                            vec_idx = dcy * kernel_offset - ((dcy * (dcy + 1)) >> 1) + dcx;
                        } else {
                            vec_idx = dcx * kernel_offset - ((dcx * (dcx + 1)) >> 1) + dcy;
                        }

                        tmp_cache.at(tx, ty) = (*ptr)[vec_idx];
                        kernel_sum += reinterpret_cast<real_t(&)[2]>((*ptr)[vec_idx])[0];
                        tx++;
                    }
                    tx = 0;
                    ty++;
                }
                cache(y, x) = tmp_cache / kernel_sum;
            }
        }
    } else { // kernel from not shifted fft

        const size_t kernel_half_size = arr_size / 2;
        const size_t oversamp_conv = ctr_idx - max_conv_support * oversamp + oversampled_pixel;
        arma::Mat<cx_real_t>* ptr = &conv_kernel;

        for (size_t x = 0; x < cache_size; x++) {
            for (size_t y = 0; y < cache_size; y++) {

                size_t cx, cy;
                size_t xx_start = oversamp_conv - x;
                size_t yy_start = oversamp_conv - y;
                double kernel_sum = 0;
                for (size_t ty = 0; ty < tmp_cache_size; ty++, xx_start += oversamp) {

                    cx = (xx_start + kernel_half_size) % arr_size;
                    for (size_t tx = 0; tx < tmp_cache_size; tx++, yy_start += oversamp) {

                        cy = (yy_start + kernel_half_size) % arr_size;
                        tmp_cache.at(tx, ty) = (*ptr).at(cx, cy);
                        kernel_sum += reinterpret_cast<real_t(&)[2]>((*ptr).at(cx, cy))[0];
                    }

                    yy_start = oversamp_conv - y;
                }

                kernel_sum = 1 / kernel_sum;
                cache(y, x) = tmp_cache * kernel_sum;
            }
        }
    }
    STPLIB_DEBUG(spdlog::get("stplib"), "W-Proj: Kernel maximum = {}, minimum = {}", std::abs(cache(oversampling / 2, oversampling / 2).at(cache(oversampling / 2, oversampling / 2).n_rows / 2, cache(oversampling / 2, oversampling / 2).n_cols / 2)), std::abs(cache(oversampling / 2, oversampling / 2).at(0, cache(oversampling / 2, oversampling / 2).n_cols / 2)));
    return std::move(cache);
}

MatStp<cx_real_t> WideFieldImaging::combine_kernels(const arma::Col<real_t>& aa_kernel_img, const double scaled_cell_size)
{
    /* kernel:
 *      +---- ... ----
 *      -xxxx ... xxxx
 *      -xxxx ... xxxx
 *      -xxxx ... xxxx
 *      -xxxx ... xxxx
 *       ...  ... ...
 *      -xxxx ... xxxx
 *      -xxxx ... xxxx
 *      -xxxx ... xxxx
 *      -xxxx ... xxxx
 *
 * + points
 * - lines
 * x squares
 */
    int arr_size = array_size;
    double fft_factor = 1.0 / (kernel_size * kernel_size);
    size_t half_kernel_size = kernel_size / 2;

    MatStp<cx_real_t> output(arr_size, arr_size);

    // calculate lines
    tbb::parallel_for(tbb::blocked_range<size_t>(1, half_kernel_size), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
            cx_real_t value = combine_2d_kernel_value(aa_kernel_img, i, 0, fft_factor, w_value, scaled_cell_size, half_kernel_size);
            output.at(i, 0) = value;
            output.at(arr_size - i, 0) = value;
            output.at(0, i) = value;
            output.at(0, arr_size - i) = value;
        }
    });

    // calculate points
    {
        // max*max
        output.at(0, 0) = aa_kernel_img.at(half_kernel_size) * aa_kernel_img.at(half_kernel_size) * fft_factor;
    }

    // calculate squares
    tbb::parallel_for(tbb::blocked_range<size_t>(1, half_kernel_size),
        [&](const tbb::blocked_range<size_t>& r) {
            size_t j_begin = r.begin(), j_end = r.end();
            for (size_t j = j_begin; j < j_end; ++j) {
                for (size_t i = 1; i < half_kernel_size; ++i) {
                    cx_real_t value = combine_2d_kernel_value(aa_kernel_img, i, j, fft_factor, w_value, scaled_cell_size, half_kernel_size);
                    output.at(i, j) = value;
                    output.at(arr_size - i, j) = value;
                    output.at(i, arr_size - j) = value;
                    output.at(arr_size - i, arr_size - j) = value;
                }
            }
        });

    return std::move(output);
}

/* Calculate kernel value by combining w_kernel and aa_img_domain_kernel */
inline cx_real_t WideFieldImaging::combine_2d_kernel_value(const arma::Col<real_t>& aa_kernel_img,
    const int pixel_x, const int pixel_y, const real_t fft_factor,
    const double w_value, const double scaled_cell_size, const int ctr_idx)
{
    real_t distance_x, distance_y, squared_radians, n;
    distance_x = double(pixel_x) * scaled_cell_size;
    distance_y = double(pixel_y) * scaled_cell_size;
    cx_real_t value;

    squared_radians = distance_x * distance_x + distance_y * distance_y;

    if (squared_radians > 1.0) {
        value = aa_kernel_img.at(ctr_idx + pixel_x) * aa_kernel_img.at(ctr_idx + pixel_y) * fft_factor;
    } else {
        n = std::sqrt(1.0 - squared_radians);
        cx_real_t im(0, (-2.0 * M_PI * w_value * (n - 1.0)));
        value = (std::exp(im) / n) * aa_kernel_img.at(ctr_idx + pixel_x) * aa_kernel_img.at(ctr_idx + pixel_y) * fft_factor;
    }
    return value;
}

/* generate Hankel radius diagonal points */
arma::Col<real_t> WideFieldImaging::generate_hankel_radius_points(size_t arrsize)
{
    arma::Col<real_t> r_points(arrsize);
    for (size_t i = 0; i < arrsize; i++) {
        r_points(i) = real_t(i) * M_SQRT1_2;
    }

    return std::move(r_points);
}

/*************************************/
/* Bessel function kind 1, order 1   */
/*************************************/
inline real_t besselj1(real_t x)
{
    real_t z, xx, y, res, tmp1, tmp2;

    if (x < 8.0) {
        xx = x * x;
        tmp1 = x * (72362614232.0 + xx * (-7895059235.0 + xx * (242396853.1 + xx * (-2972611.439 + xx * (15704.48260 + xx * (-30.16036606))))));
        tmp2 = 144725228442.0 + xx * (2300535178.0 + xx * (18583304.74 + xx * (99447.43394 + xx * (376.9991397 + xx))));
        res = tmp1 / tmp2;
    } else {
        z = 8.0 / x;
        xx = z * z;
        y = x - 2.356194491;
        tmp1 = 1.0 + xx * (0.183105e-2 + xx * (-0.3516396496e-4 + xx * (0.2457520174e-5 + xx * (-0.240337019e-6))));
        tmp2 = 0.04687499995 + xx * (-0.2002690873e-3 + xx * (0.8449199096e-5 + xx * (-0.88228987e-6 + xx * 0.105787412e-6)));
        res = sqrt(0.636619772 / x) * (cos(y) * tmp1 - z * sin(y) * tmp2);
    }
    return res;
}

/* Generate Discrete Hankel Transform (DHT) */
// Note that DHT matrix is generated transposed for faster dot product during Hankel transform
arma::Mat<real_t> WideFieldImaging::dht(size_t arrsize)
{
    // aux vars
    arma::Col<real_t> k(arrsize);
    arma::Col<real_t> rn(arrsize);
    arma::Col<real_t> kn(arrsize);
    arma::Mat<real_t> DHT(arrsize, arrsize);

    real_t kf = M_PI / real_t(arrsize);
    kn[0] = 0.0;
    k[0] = 0.0;
    rn[0] = 0.5;
    tbb::parallel_for(tbb::blocked_range<size_t>(1, arrsize), [&](const tbb::blocked_range<size_t>& r) {
        size_t i_end = r.end();

        for (size_t i = r.begin(); i < i_end; ++i) {
            real_t idx = i;
            k[i] = kf * idx;
            rn[i] = idx + 0.5;
            kn[i] = 2.0 * M_PI / k[i];
        }
    });

    // last rn value is
    rn.at(arrsize - 1) = arrsize - 1;

    /* generate Discrete Hankel Transform (DHT) */
    // fill first I value
    DHT.at(0, 0) = M_PI * rn(0) * rn(0);

    // diferential vars
    double prev_value = DHT.at(0, 0);
    double cur_value;

    // fill first I line
    for (size_t j = 1; j < arrsize; j++) {
        cur_value = M_PI * rn[j] * rn[j];
        DHT.at(j, 0) = cur_value - prev_value;
        prev_value = cur_value;
    }

    //calculate remaining rows
    tbb::parallel_for(tbb::blocked_range<size_t>(1, arrsize),
        [&](const tbb::blocked_range<size_t>& r) {
            size_t col_begin = r.begin(), col_end = r.end();
            if (col_begin == 0)
                col_begin += 1;

            for (size_t i = col_begin; i < col_end; ++i) {
                const real_t& kn_i = kn[i];
                const real_t& k_i = k[i];

                const real_t& rn_j = rn[arrsize - 1];
                DHT.at(arrsize - 1, i) = kn_i * rn_j * besselj1(k_i * rn_j);

                for (size_t j = arrsize - 1; j > 0; --j) {
                    const real_t& rn_j = rn[j - 1];
                    DHT.at(j - 1, i) = kn_i * rn_j * besselj1(k_i * rn_j);
                    DHT.at(j, i) -= DHT.at(j - 1, i);
                }
            }
        });

    return std::move(DHT);
}

// Local functions

double parangle(double ha, double dec_d, double lat_d)
{
    double lat_r = lat_d * M_PI / 180.0;
    double dec_r = dec_d * M_PI / 180.0;
    double ha_r = ha * 15.0 * M_PI / 180.0;
    double sin_eta_sin_z = cos(lat_r) * sin(ha_r);
    double cos_eta_sin_z = sin(lat_r) * cos(dec_r) - cos(lat_r) * sin(dec_r) * cos(ha_r);
    double eta = atan2(sin_eta_sin_z, cos_eta_sin_z);

    return eta;
}
}
