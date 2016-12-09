/** @file conv_func.h
 *  @brief Classes and functions of convolution methods
 *
 *  This contains the prototypes and the classes for the convolution
 *  functions
 *
 *  @bug No known bugs.
 */

#ifndef CONV_FUNC_H
#define CONV_FUNC_H

#include <armadillo>
#include <complex>
#include <utility>

const double oversampling_disabled(-100);
const double tolerance(0.000000000000002);

/**
 * @brief The TopHat functor class
 */
class TopHat {
public:
    /**
     * @brief TopHat constructor
     * @param half_base_width (double)
     */
    TopHat(double half_base_width)
        : _half_base_width(half_base_width)
    {
    }

    /**
     * @brief operator ()
     * @param radius_in_pix (arma::mat)
     * @return TopHat mat
     */
    arma::mat operator()(const arma::mat& radius_in_pix) const;

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
     * @param[in] triangle_value (double)
     */
    Triangle(double half_base_width, double triangle_value)
        : _half_base_width(half_base_width)
        , _triangle_value(triangle_value)
    {
    }

    /**
     * @brief operator ()
     * @param[in] radius_in_pix (arma::mat)
     * @return Triangle mat
     */
    arma::mat operator()(const arma::mat& radius_in_pix) const;

private:
    const double _half_base_width;
    const double _triangle_value;
};

/**
 * @brief The Sinc functor class
 */
class Sinc {
public:
    /** @brief Default constructor, with no truncation.
     */
    Sinc() = default;

    /**
     * @brief Constructor with no truncation.
     * @param[in] width_normalization (double)
     */
    Sinc(double width_normalization)
        : _width_normalization(width_normalization)
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
     * @param[in] radius_in_pix (arma::mat&)
     * @return Convolution kernel
     */
    arma::mat operator()(const arma::mat& radius_in_pix) const;

private:
    bool _truncate = false;
    double _threshold = 0.0;

    double _width_normalization = 1.0;
};

/**
 * @brief The Gaussian functor class
 */
class Gaussian {
public:
    /**
     * @brief Default constructor
     */
    Gaussian() = default;

    /**
     * @brief Constructor with no truncation.
     * @param[in] width_normalization (double)
     */
    Gaussian(double width_normalization)
        : _width_normalization(width_normalization)
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
     * @param[in] radius_in_pix (arma:mat&)
     * @return Convolution kernel
     */
    arma::mat operator()(const arma::mat& radius_in_pix) const;

private:
    bool _truncate = false;
    double _threshold = 0.0;

    double _width_normalization = 1.0;
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
        : _gaussian(_default_width_normalization_gaussian)
        , _sinc(_default_width_normalization_sinc)
    {
    }

    /**
     * @brief Constructor with no truncation.
     * @param[in] width_normalization_gaussian (double)
     * @param[in] width_normalization_sinc (double)
     */
    GaussianSinc(double width_normalization_gaussian, double width_normalization_sinc)
        : _gaussian(width_normalization_gaussian)
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
     * @param[in] radius_in_pix (arma:mat&)
     * @return Convolution kernel
     */
    arma::mat operator()(const arma::mat& radius_in_pix) const;

private:
    const bool _truncate = false;
    const double _threshold = 0.0;

    Gaussian _gaussian;
    Sinc _sinc;

    const double _default_width_normalization_gaussian = 2.52;
    const double _default_width_normalization_sinc = 1.55;
};

/** @brief Make Kernel Array
*
*  Function (template + functor) to create a Kernel array with some specs.
*
*  @param[in] support (int): Defines the 'radius' of the bounding box within
*              which convolution takes place. `Box width in pixels = 2*support+1`.
*              (The central pixel is the one nearest to the UV co-ordinates.)
*              For a kernel_func with truncation radius `trunc`, the support
*              should be set to `ceil(trunc+0.5)` to ensure that the kernel
*              function is fully supported for all valid subpixel offsets.
*
*  @param[in] offset (arma::mat): 2-vector subpixel offset from the sampling position of the
*              central pixel to the origin of the kernel function.
*
*  @param[in] pad (bool) : Whether to pad the array by an extra pixel-width.
*             This is used when generating an oversampled kernel that will be used for interpolation.
*  @param[in] normalize (bool)
*
*  @return Result kernel
*/
template <typename T>
arma::mat make_kernel_array(int support, const arma::mat& offset, double oversampling, bool pad, bool normalize, const T& kernel_creator)
{

    int localOversampling = (oversampling < oversampling_disabled || (oversampling > oversampling_disabled)) ? oversampling : 1.0;
    int localPad = (pad == true) ? 1.0 : 0.0;

    int array_size = 2 * (support + localPad) * localOversampling + 1;
    int centre_idx = (support + pad) * localOversampling;

    arma::mat distance_vec((arma::linspace(0, array_size - 1, array_size) - centre_idx) / localOversampling);

    // Call the functor's operator ()
    arma::vec x_kernel_coeffs = kernel_creator(distance_vec - offset[0]);
    arma::vec y_kernel_coeffs = kernel_creator(distance_vec - offset[1]);

    // Multiply the two vectors obtained with convolution function to obtain the 2D kernel.
    arma::mat result = arma::repmat(y_kernel_coeffs, 1, array_size) * arma::diagmat(x_kernel_coeffs);

    return (normalize == true) ? (result / arma::accu(result)) : result;
}

#endif /* CONV_FUNC_H */
