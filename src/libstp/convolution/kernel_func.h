/** @file kernel_func.h
 *  @brief Kernel functions
 *
 *  This contains the function for the 2D kernel convolution
 *
 *  @bug No known bugs.
 */

#ifndef KERNEL_FUNC_H
#define KERNEL_FUNC_H

#include "conv_func.h"

/** @brief Make Kernel Array
*
*	Function to create a Kernel array with some specs
*
* 	@param[in] support (int): Defines the 'radius' of the bounding box within
*              which convolution takes place. `Box width in pixels = 2*support+1`.
*              (The central pixel is the one nearest to the UV co-ordinates.)
*              For a kernel_func with truncation radius `trunc`, the support
*              should be set to `ceil(trunc+0.5)` to ensure that the kernel
*              function is fully supported for all valid subpixel offsets.
*
* 	@param[in] offset (arma::mat): 2-vector subpixel offset from the sampling position of the<
*              central pixel to the origin of the kernel function.
*              Ordering is (x_offset,y_offset). Should have values such that `abs(offset) <= 0.5`
*              otherwise the nearest integer grid-point would be different!
*
* 	@param[in] pad (double)
*
* 	@param[in] normalize (bool)
*
*  	@return Result kernel
*/
template <typename T, typename... Args>
arma::mat  make_kernel_array(const int support, const arma::mat& offset = { 0.0, 0.0 }, double oversampling = false, double pad = false, bool normalize = false, Args&&... args) {
    T obj;

    if (oversampling == false) {
        oversampling = 1.0;
    }

    pad = (pad) ? 1.0 : 0.0;
    int array_size = 2 * (support + pad) * oversampling + 1;
    int centre_idx = (support + pad) * oversampling;
    arma::mat distance_vec = ((arma::linspace(0, array_size - 1, array_size) - centre_idx) / oversampling);

    arma::vec x_kernel_coeffs = obj((distance_vec - offset[0]), std::forward<Args>(args)...);
    arma::vec y_kernel_coeffs = obj((distance_vec - offset[1]), std::forward<Args>(args)...);

    arma::mat result = arma::repmat(y_kernel_coeffs, 1, array_size) * arma::diagmat(x_kernel_coeffs);

    if (normalize == true) {
        return result /= arma::accu(result);
    }

    return result;
}

#endif /* KERNEL_FUNC_H */
