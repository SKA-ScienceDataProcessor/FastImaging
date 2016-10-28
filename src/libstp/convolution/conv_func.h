/** @file cov_func.h
 *  @brief Classes and functions of convolution methods
 *
 *  This contains the prototypes and the classes for the convolutions
 *  functions
 *
 *  @bug No known bugs.
 */

#ifndef CONV_FUNC_H
#define CONV_FUNC_H

#include <armadillo>
#include <utility>

const double NO_OVERSAMPLING(-100);
const double tolerance(0.002);

/** @brief Class Tophat
 *
 *  Generate Tophat convolution with an array input or scalar input.
 *
 *  @param[in] radius_in_pix (arma::mat)
 *  @param[in] half_base_width (double)
 *
 *  @return Result convolution
 */
class TopHat {
    public:
        arma::mat operator() (const arma::mat& radius_in_pix, const double half_base_width);
};

/** @brief Class Triangle
 *
 *  Generate Triangle convolution with an array input or scalar input.
 *
 *  @param[in] radius_in_pix (arma::mat)
 *  @param[in] half_base_width (double)
 *  @param[in] triangle_value (double)
 *
 *  @return Result convolution
 */
class Triangle {
    public:
        arma::mat operator() (const arma::mat& radius_in_pix, const double half_base_width, const double triangle_value);
};

/** @brief Class Sinc
 *
 *  Generate Sinc convolution with an array input or scalar input.
 *
 *  @param[in] radius_in_pix (arma::mat)
 *
 *  @return Result convolution
 */
class Sinc {
    public:
        arma::mat operator() (const arma::mat& radius_in_pix);
};

/** @brief Class Gaussian
 *
 *  Generate Gaussian convolution with an array input or scalar input.
 *
 *  @param[in] radius_in_pix (arma::mat)
 *  @param[in] width (double)
 *
 *  @return Result convolution
 */
class Gaussian {
    public:
        arma::mat operator() (const arma::mat& radius_in_pix, const double width);
};

/** @brief Class GaussianSinc
 *
 *  Generate Gaussian Sinc convolution  with an array input or scalar input.
 *
 *  @param[in] radius_in_pix (arma::mat)
 *  @param[in] width_normalization_gaussian (double)
 *  @param[in] width_normalization_sinc (double)
 *
 *  @return Result convolution
 */
class GaussianSinc {
    public:
        arma::mat operator() (const arma::mat& radius_in_pix, const double width_normalization_gaussian, const double width_normalization_sinc);
};

/** @brief Make Kernel Array
*
*	Function to create a Kernel array with some specs
*
*  @param[in] support (int): Defines the 'radius' of the bounding box within
*              which convolution takes place. `Box width in pixels = 2*support+1`.
*              (The central pixel is the one nearest to the UV co-ordinates.)
*              For a kernel_func with truncation radius `trunc`, the support
*              should be set to `ceil(trunc+0.5)` to ensure that the kernel
*              function is fully supported for all valid subpixel offsets.
*
*  @param[in] offset (arma::mat): 2-vector subpixel offset from the sampling position of the<
*              central pixel to the origin of the kernel function.
*              Ordering is (x_offset,y_offset). Should have values such that `abs(offset) <= 0.5`
*              otherwise the nearest integer grid-point would be different!
*
*  @param[in] pad (bool)
*
*  @param[in] normalize (bool)
*
*  	@return Result kernel
*/
template <typename T, typename... Args>
arma::mat make_kernel_array(const int support, const arma::mat& offset, const double oversampling, const bool pad, const bool normalize, Args&&... args) {

    int localOversampling = 1.0;
    int localPad = 0.0;

    if (oversampling != NO_OVERSAMPLING) {
        localOversampling = oversampling;
    }

    if (pad == true) {
        localPad = 1.0;
    }
    
    int array_size = 2 * (support + localPad) * localOversampling + 1;
    int centre_idx = (support + pad) * localOversampling;
    arma::mat distance_vec = ((arma::linspace(0, array_size - 1, array_size) - centre_idx) / localOversampling);

    T obj;
    arma::vec x_kernel_coeffs = obj((distance_vec - offset[0]), std::forward<Args>(args)...);
    arma::vec y_kernel_coeffs = obj((distance_vec - offset[1]), std::forward<Args>(args)...);

    arma::mat result = arma::repmat(y_kernel_coeffs, 1, array_size) * arma::diagmat(x_kernel_coeffs);

    if (normalize == true) {
        result /= result;
    }

    return result;
}

#endif /* CONV_FUNC_H */
