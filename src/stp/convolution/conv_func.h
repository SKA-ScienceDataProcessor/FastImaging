/**
 * @file conv_func.h
 * @brief Classes and function prototypes of convolution methods.
 */

#ifndef CONV_FUNC_H
#define CONV_FUNC_H

#include "../common/fft.h"
#include "../types.h"

#include <armadillo>
#include <cassert>
#include <complex>
#include <tbb/tbb.h>
#include <utility>

namespace stp {

/**
 * @brief The TopHat functor class
 */
class TopHat {
public:
    /**
     * @brief TopHat constructor
     * @param[in] half_base_width (double)
     */
    TopHat(const double half_base_width)
        : _half_base_width(half_base_width)
    {
    }

    /**
     * @brief Operator ()
     * @param[in] radius_in_pix (arma::Col<real_t>)
     * @return TopHat mat
     */
    arma::Col<real_t> operator()(const arma::Col<real_t>& radius_in_pix) const;

    /**
     * @brief Generates the 1D grid correction function (gcf)
     * @param[in] radius (arma::Col<real_t>&)
     * @return 1D grid correction function (gcf)
     */
    arma::Col<real_t> gcf(const arma::Col<real_t>& radius) const;

private:
    double _half_base_width;
};

/**
 * @brief The Triangle functor class
 */
class Triangle {
public:
    /**
     * @brief Triangle constructor
     *
     * Generate Triangle convolution with an array input or scalar input.
     *
     * @param[in] half_base_width (double)
     */
    Triangle(const double half_base_width)
        : _half_base_width(half_base_width)
    {
    }

    /**
     * @brief operator ()
     * @param[in] radius_in_pix (arma::Col<real_t>)
     * @return Triangle mat
     */
    arma::Col<real_t> operator()(const arma::Col<real_t>& radius_in_pix) const;

    /**
     * @brief Generates the 1D grid correction function (gcf)
     * @param[in] radius (arma::Col<real_t>&)
     * @return 1D grid correction function (gcf)
     */
    arma::Col<real_t> gcf(const arma::Col<real_t>& radius) const;

private:
    const double _half_base_width;
};

/**
 * @brief The Sinc functor class
 */
class Sinc {
public:
    /** @brief Default constructor, with no truncation.
     */
    Sinc()
        : _trunc(0.0)
        , _width_normalization(1.0)
    {
    }

    /**
     * @brief Sinc constructor with truncation threshold.
     * @param[in] trunc (double) Truncation radius.
     */
    Sinc(const double trunc, double width_normalization = 1.0)
        : _trunc(trunc)
        , _width_normalization(width_normalization)
    {
    }

    /**
     * @brief operator ()
     * @param[in] radius_in_pix (arma::Col<real_t>&)
     * @return Convolution kernel
     */
    arma::Col<real_t> operator()(const arma::Col<real_t>& radius_in_pix) const;

    /**
     * @brief Generates the 1D grid correction function (gcf)
     * @param[in] radius (arma::Col<real_t>&)
     * @return 1D grid correction function (gcf)
     */
    arma::Col<real_t> gcf(const arma::Col<real_t>& radius) const;

private:
    double _trunc;
    double _width_normalization;
};

/**
 * @brief The Gaussian functor class
 */
class Gaussian {
public:
    /**
     * @brief Default constructor
     */
    Gaussian()
        : _trunc(0.0)
        , _width_normalization(1.0)
    {
    }

    /**
     * @brief Constructor with truncation
     * @param[in] trunc (double) Truncation radius.
     * @param[in] width_normalization (double)
     */
    Gaussian(const double trunc, const double width_normalization = 1.0)
        : _trunc(trunc)
        , _width_normalization(width_normalization)
    {
    }

    /**
     * @brief operator ()
     * @param[in] radius_in_pix (arma::Col<real_t>&)
     * @return Convolution kernel
     */
    arma::Col<real_t> operator()(const arma::Col<real_t>& radius_in_pix) const;

    /**
     * @brief Generates the 1D grid correction function (gcf)
     * @param[in] radius (arma::Col<real_t>&)
     * @return 1D grid correction function (gcf)
     */
    arma::Col<real_t> gcf(const arma::Col<real_t>& radius) const;

private:
    double _trunc;
    double _width_normalization;
};

/**
 * @brief The GaussianSinc functor class
 */
class GaussianSinc {
public:
    /**
     * @brief Default constructor
     */
    GaussianSinc()
        : _trunc(0.0)
        , _gaussian(0.0, _default_width_normalization_gaussian)
        , _sinc(0.0, _default_width_normalization_sinc)
    {
    }

    /**
     * @brief GaussianSinc Constructor with default parameters and truncation.
     * @param[in] trunc (double) Truncation radius.
     * @param[in] width_normalization_gaussian (double)
     * @param[in] width_normalization_sinc (double)
     */
    GaussianSinc(const double trunc, const double width_normalization_gaussian = _default_width_normalization_gaussian, const double width_normalization_sinc = _default_width_normalization_sinc)
        : _trunc(trunc)
        , _gaussian(0.0, width_normalization_gaussian)
        , _sinc(0.0, width_normalization_sinc)
    {
    }

    /**
     * @brief operator ()
     * @param[in] radius_in_pix (arma::Col<real_t>&)
     * @return Convolution kernel
     */
    arma::Col<real_t> operator()(const arma::Col<real_t>& radius_in_pix) const;

    /**
     * @brief Generates the 1D grid correction function (gcf)
     * @param[in] radius (arma::Col<real_t>&)
     * @return 1D grid correction function (gcf)
     */
    arma::Col<real_t> gcf(const arma::Col<real_t>& radius) const;

private:
    static constexpr double _default_width_normalization_gaussian = 2.52;
    static constexpr double _default_width_normalization_sinc = 1.55;
    const double _trunc;

    Gaussian _gaussian;
    Sinc _sinc;
};

/**
 * @brief The Prolate spheroidal wave function (PSWF) functor class
 */
class PSWF {
public:
    /**
     * @brief Default constructor
     */
    PSWF()
        : _trunc(0.0)
    {
    }

    /**
     * @brief Constructor with truncation
     * @param[in] trunc (double) Truncation radius.
     */
    PSWF(const double trunc)
        : _trunc(trunc)
    {
    }

    /**
     * @brief operator ()
     * @param[in] radius_in_pix (arma::Col<real_t>&)
     * @return Convolution kernel
     */
    arma::Col<real_t> operator()(const arma::Col<real_t>& radius_in_pix) const;

    /**
     * @brief Generates the 1D grid correction function (gcf)
     * @param[in] radius (arma::Col<real_t>&)
     * @return 1D grid correction function (gcf)
     */
    arma::Col<real_t> gcf(const arma::Col<real_t>& radius) const;

private:
    double _trunc;
};

/** @brief Make 2D Kernel Array
*
*  Function (template + functor) to create a Kernel array with some specs.
*
*  @param[in] kernel_creator: functor used for kernel generation
*  @param[in] support (int): Defines the 'radius' of the bounding box within which convolution takes place.
*  @param[in] offset (arma::Col<real_t>): 2-vector subpixel offset from the sampling position of the
*                                central pixel to the origin of the kernel function.
*  @param[in] oversampling (int): Controls kernel-generation.
*  @param[in] pad (bool): Whether to pad the array by an extra pixel-width. This is used when generating an
*                          oversampled kernel that will be used for interpolation.
*  @param[in] normalize (bool): Whether or not the returned image should be normalized
*
*  @return Result kernel
*/
template <typename T>
arma::Mat<real_t> make_kernel_array(const T& kernel_creator, int support, const arma::mat& offset, int oversampling = 1, bool pad = false, bool normalize = true)
{
    assert(support >= 1);
    assert(offset.n_elem == 2);
    assert(fabs(offset[0]) <= 0.5);
    assert(fabs(offset[1]) <= 0.5);
    assert(oversampling >= 1);

    int localPad = (pad == true) ? 1 : 0;

    int array_size = 2 * (support + localPad) * oversampling + 1;
    int centre_idx = (support + localPad) * oversampling;

    arma::Col<real_t> distance_vec((arma::linspace<arma::Col<real_t>>(0, array_size - 1, array_size) - centre_idx) / oversampling);

    // Call the functor's operator ()
    arma::Col<real_t> x_kernel_coeffs = kernel_creator(distance_vec - offset[0]);
    arma::Col<real_t> y_kernel_coeffs = kernel_creator(distance_vec - offset[1]);

    // Multiply the two vectors obtained with convolution function to obtain the 2D kernel.
    arma::Mat<real_t> result = y_kernel_coeffs * x_kernel_coeffs.st();

    return (normalize == true) ? (result / arma::accu(result)) : result;
}

/** @brief Make 1D Kernel Array
*
*  Function (template + functor) to create 1D Kernel array with some specs.
*
*  @param[in] kernel_creator: functor used for kernel generation
*  @param[in] support (int): Defines the 'radius' of the bounding box within which convolution takes place.
*  @param[in] offset (double): subpixel offset from the sampling position of the central pixel to the origin of the kernel function.
*  @param[in] oversampling (int): Controls kernel-generation.
*  @param[in] pad (bool): Whether to pad the array by an extra pixel-width. This is used when generating an
*                          oversampled kernel that will be used for interpolation.
*  @param[in] normalize (bool): Whether or not the returned image should be normalized
*
*  @return (arma::Col<real_t>) Result 1D kernel
*/
template <typename T>
arma::Col<real_t> make_1D_kernel(const T& kernel_creator, int support, const double offset, int oversampling = 1, bool pad = false, bool normalize = true)
{
    assert(support >= 1);
    assert(fabs(offset) <= 0.5);
    assert(oversampling >= 1);

    int localPad = (pad == true) ? 1 : 0;

    int array_size = 2 * (support + localPad) * oversampling + 1;
    int centre_idx = (support + localPad) * oversampling;

    arma::Col<real_t> distance_vec((arma::linspace<arma::Col<real_t>>(0, array_size - 1, array_size) - centre_idx) / oversampling);

    // Call the functor's operator ()
    arma::Col<real_t> result = kernel_creator(distance_vec - offset);

    return (normalize == true) ? (result / arma::accu(result)) : result;
}

/**
 * @brief Computes the image-domain 1D kernel using the forward fast fourier transform or the analytic definition
 *
 *  Function (template + functor) to create 1D Kernel array with some specs.
 *
 * @param[in] kernel_creator: functor used for kernel generation
 * @param[in] kernel_size (size_t): Defines the array size.
 * @param[in] normalize (bool): Whether to normalize output or not.
 * @param[in] analytic_gfc (bool): Use analytic definition if true.
 * @param[in] r_fft (FFTRoutine): Selects FFT routine to be used.
 * @return (arma::Col<real_t>): 1D array of the image domain kernel
 */
template <typename T>
arma::Col<real_t> ImgDomKernel(const T& kernel_creator, size_t kernel_size, bool normalize = false, bool analytic_gcf = false, FFTRoutine r_fft = FFTRoutine::FFTW_ESTIMATE_FFT)
{
    arma::Col<real_t> aa_kernel_img(kernel_size); // FFT 1D kernel complete array

    size_t centre_idx = kernel_size / 2;

    if (analytic_gcf == true) {
        /* Create 1D kernel */
        aa_kernel_img = kernel_creator.gcf((arma::linspace<arma::Col<real_t>>(0, kernel_size - 1, kernel_size) - centre_idx) / centre_idx);
    } else {
        arma::Col<real_t> kernel1D_array(kernel_size); //  1D kernel array
        arma::Col<real_t> fft_kernel1D_array(kernel_size); //  1D kernel array

        /* Create 1D kernel */
        arma::Col<real_t> kernel1D = kernel_creator(arma::linspace<arma::Col<real_t>>(0, kernel_size - 1, kernel_size) - centre_idx);

        // normalize
        double kernel_sum = arma::accu((kernel1D));
        if (kernel_sum > 0.0)
            kernel1D = kernel1D / kernel_sum;

        /*** Generate 1D kernel array */
        // set set kernel array to zeros
        kernel1D_array.zeros();
        for (size_t i = 0; i < centre_idx; i++) {
            kernel1D_array(i) = kernel1D(centre_idx + i);
            kernel1D_array(kernel_size - centre_idx + i) = kernel1D(i);
        }

        // dft the kernel 1D array
        if (r_fft == FFTRoutine::FFTW_WISDOM_FFT || r_fft == FFTRoutine::FFTW_WISDOM_INPLACE_FFT) {
            if (kernel_size < 4) {
                r_fft = FFTRoutine::FFTW_ESTIMATE_FFT;
            }
        }
        // dft the kernel 1D array
        fft_fftw_dft_r2r_1d(kernel1D_array, fft_kernel1D_array, r_fft);

        // invert and duplicate array values
        aa_kernel_img.zeros();

        tbb::parallel_for(tbb::blocked_range<size_t>(0, kernel_size / 2), [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                aa_kernel_img[kernel_size / 2 + i] = fft_kernel1D_array[i];
            }
        });
        tbb::parallel_for(tbb::blocked_range<size_t>(1, kernel_size / 2 + 1), [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i < r.end(); ++i) {
                aa_kernel_img[kernel_size / 2 - i] = fft_kernel1D_array[i];
            }
        });
    }

    if (normalize == true) {
        double kernel_sum = arma::accu((aa_kernel_img));
        if (kernel_sum > 0.0)
            aa_kernel_img = aa_kernel_img / kernel_sum;
    }

    return std::move(aa_kernel_img);
}
}

#endif /* CONV_FUNC_H */
