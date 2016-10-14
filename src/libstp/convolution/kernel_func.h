/** @file kernel_func.h
 *  @brief Kernel function prototypes functions
 *
 *  This contains the prototypes for the kernel
 *
 *  @bug No known bugs.
 */

#ifndef KERNEL_FUNC_H
#define KERNEL_FUNC_H

#include "conv_func.h"


/** @brief Construct TopHat Kernel Array Function
*
*	Function to create a TopHat Kernel array with some specs
*
* 	@param[in] kernel_func (func): Callable object that returns a convolution
*              co-efficient for a given distance in pixel-widths.
*
* 	@param[in] support (int): Defines the 'radius' of the bounding box within
*              which convolution takes place. `Box width in pixels = 2*support+1`.
*              (The central pixel is the one nearest to the UV co-ordinates.)
*              For a kernel_func with truncation radius `trunc`, the support
*              should be set to `ceil(trunc+0.5)` to ensure that the kernel
*              function is fully supported for all valid subpixel offsets.
*
* 	@param[in] offset (mat): 2-vector subpixel offset from the sampling position of the
*              central pixel to the origin of the kernel function.
*              Ordering is (x_offset,y_offset). Should have values such that `abs(offset) <= 0.5`
*              otherwise the nearest integer grid-point would be different!
*
* 	@param[in] oversampling (int)
*
* 	@param[in] half_base_width (double)
*
*  	@return Result kernel
*/
mat make_top_hat_kernel_array(const int support, const mat& offset, const double oversampling, const double half_base_width);

/** @brief Construct Triangle Kernel Array Function
*
*	Function to create a Triangle Kernel array with some specs
*
* 	@param[in] kernel_func (func): Callable object that returns a convolution
*              co-efficient for a given distance in pixel-widths.
*
* 	@param[in] support (int): Defines the 'radius' of the bounding box within
*              which convolution takes place. `Box width in pixels = 2*support+1`.
*              (The central pixel is the one nearest to the UV co-ordinates.)
*              For a kernel_func with truncation radius `trunc`, the support
*              should be set to `ceil(trunc+0.5)` to ensure that the kernel
*              function is fully supported for all valid subpixel offsets.
*
* 	@param[in] offset (mat): 2-vector subpixel offset from the sampling position of the
*              central pixel to the origin of the kernel function.
*              Ordering is (x_offset,y_offset). Should have values such that `abs(offset) <= 0.5`
*              otherwise the nearest integer grid-point would be different!
*
* 	@param[in] oversampling (int)
*
* 	@param[in] half_base_width (double)
*
*  	@return Result kernel
*/
mat make_triangle_kernel_array(const int support, const mat& offset, const double oversampling, const double half_base_width);

#endif /* KERNEL_FUNC_H */
