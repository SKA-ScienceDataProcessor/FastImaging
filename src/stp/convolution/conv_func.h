/**
 * @file conv_func.h
 * @brief Classes and function prototypes of convolution methods.
 */

#ifndef CONV_FUNC_H
#define CONV_FUNC_H

#include <armadillo>
#include <cassert>
#include <complex>
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
    TopHat(double half_base_width)
        : _half_base_width(half_base_width)
    {
    }

    /**
     * @brief Operator ()
     * @param[in] radius_in_pix (arma::vec)
     * @return TopHat mat
     */
    arma::vec operator()(const arma::vec& radius_in_pix) const;

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
    Triangle(double half_base_width)
        : _half_base_width(half_base_width)
    {
    }

    /**
     * @brief operator ()
     * @param[in] radius_in_pix (arma::vec)
     * @return Triangle mat
     */
    arma::vec operator()(const arma::vec& radius_in_pix) const;

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
        : _truncate(false)
        , _threshold(0.0)
        , _width_normalization(1.0)
    {
    }

    /**
     * @brief Constructor with no truncation.
     * @param[in] width_normalization (double)
     */
    Sinc(double width_normalization)
        : _truncate(false)
        , _threshold(0.0)
        , _width_normalization(width_normalization)
    {
    }

    /**
     * @brief Sinc constructor with truncation threshold.
     * @param[in] truncation threshold
     */
    Sinc(double width_normalization, const double threshold)
        : _truncate(true)
        , _threshold(threshold)
        , _width_normalization(width_normalization)
    {
    }

    /**
     * @brief operator ()
     * @param[in] radius_in_pix (arma::vec&)
     * @return Convolution kernel
     */
    arma::vec operator()(const arma::vec& radius_in_pix) const;

private:
    bool _truncate;
    double _threshold;
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
        : _truncate(false)
        , _threshold(0.0)
        , _width_normalization(1.0)
    {
    }

    /**
     * @brief Constructor with no truncation.
     * @param[in] width_normalization (double)
     */
    Gaussian(double width_normalization)
        : _truncate(false)
        , _threshold(0.0)
        , _width_normalization(width_normalization)
    {
    }

    /**
     * @brief Constructor with truncation
     * @param[in] width_normalization (double)
     * @param[in] threshold (double)
     */
    Gaussian(const double width_normalization, double threshold)
        : _truncate(true)
        , _threshold(threshold)
        , _width_normalization(width_normalization)
    {
    }

    /**
     * @brief operator ()
     * @param[in] radius_in_pix (arma::vec&)
     * @return Convolution kernel
     */
    arma::vec operator()(const arma::vec& radius_in_pix) const;

private:
    bool _truncate;
    double _threshold;
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
        : _truncate(false)
        , _threshold(0.0)
        , _gaussian(_default_width_normalization_gaussian)
        , _sinc(_default_width_normalization_sinc)
    {
    }

    /**
     * @brief GaussianSinc Constructor with default parameters and truncation.
     * @param[in] threshold The truncation threshold.
     */
    GaussianSinc(double threshold)
        : _truncate(true)
        , _threshold(threshold)
        , _gaussian(_default_width_normalization_gaussian)
        , _sinc(_default_width_normalization_sinc)
    {
    }

    /**
     * @brief Constructor with no truncation.
     * @param[in] width_normalization_gaussian (double)
     * @param[in] width_normalization_sinc (double)
     */
    GaussianSinc(double width_normalization_gaussian, double width_normalization_sinc)
        : _truncate(false)
        , _threshold(0.0)
        , _gaussian(width_normalization_gaussian)
        , _sinc(width_normalization_sinc)
    {
    }

    /**
     * @brief Constructor with truncation threshold.
     * @param[in] width_normalization_gaussian
     * @param[in] width_normalization_sinc
     * @param[in] threshold
     */
    GaussianSinc(double width_normalization_gaussian, double width_normalization_sinc, double threshold)
        : _truncate(true)
        , _threshold(threshold)
        , _gaussian(width_normalization_gaussian)
        , _sinc(width_normalization_sinc)
    {
    }

    /**
     * @brief operator ()
     * @param[in] radius_in_pix (arma::vec&)
     * @return Convolution kernel
     */
    arma::vec operator()(const arma::vec& radius_in_pix) const;

private:
    static constexpr double _default_width_normalization_gaussian = 2.52;
    static constexpr double _default_width_normalization_sinc = 1.55;

    const bool _truncate;
    const double _threshold;

    Gaussian _gaussian;
    Sinc _sinc;
};

/** @brief Make 2D Kernel Array
*
*  Function (template + functor) to create a Kernel array with some specs.
*
*  @param[in] kernel_creator: functor used for kernel generation
*  @param[in] support (int): Defines the 'radius' of the bounding box within which convolution takes place.
*  @param[in] offset (arma::vec): 2-vector subpixel offset from the sampling position of the
*                                central pixel to the origin of the kernel function.
*  @param[in] oversampling (int): Controls kernel-generation.
*  @param[in] pad (bool): Whether to pad the array by an extra pixel-width. This is used when generating an
*                          oversampled kernel that will be used for interpolation.
*  @param[in] normalize (bool): Whether or not the returned image should be normalized
*
*  @return Result kernel
*/
template <typename T>
arma::mat make_kernel_array(const T& kernel_creator, int support, const arma::mat& offset, int oversampling = 1, bool pad = false, bool normalize = true)
{
    assert(support >= 1);
    assert(offset.n_elem == 2);
    assert(fabs(offset[0]) <= 0.5);
    assert(fabs(offset[1]) <= 0.5);
    assert(oversampling >= 1);

    int localPad = (pad == true) ? 1 : 0;

    int array_size = 2 * (support + localPad) * oversampling + 1;
    int centre_idx = (support + localPad) * oversampling;

    arma::vec distance_vec((arma::linspace(0, array_size - 1, array_size) - centre_idx) / oversampling);

    // Call the functor's operator ()
    arma::vec x_kernel_coeffs = kernel_creator(distance_vec - offset[0]);
    arma::vec y_kernel_coeffs = kernel_creator(distance_vec - offset[1]);

    // Multiply the two vectors obtained with convolution function to obtain the 2D kernel.
    arma::mat result = y_kernel_coeffs * x_kernel_coeffs.st();

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
*  @return (arma::vec) Result 1D kernel
*/
template <typename T>
arma::vec make_1D_kernel(const T& kernel_creator, int support, const double offset, int oversampling = 1, bool pad = false, bool normalize = true)
{
    assert(support >= 1);
    assert(fabs(offset) <= 0.5);
    assert(oversampling >= 1);

    int localPad = (pad == true) ? 1 : 0;

    int array_size = 2 * (support + localPad) * oversampling + 1;
    int centre_idx = (support + localPad) * oversampling;

    arma::vec distance_vec((arma::linspace(0, array_size - 1, array_size) - centre_idx) / oversampling);

    // Call the functor's operator ()
    arma::vec result = kernel_creator(distance_vec - offset);

    return (normalize == true) ? (result / arma::accu(result)) : result;
}
}

#endif /* CONV_FUNC_H */
