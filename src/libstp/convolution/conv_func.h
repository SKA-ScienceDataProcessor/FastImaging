/** @file cov_func.h
 *  @brief Function prototypes functions and classes
 *
 *  This contains the prototypes for the convolutions
 *  functions
 *
 *  @bug No known bugs.
 */

#ifndef CONV_FUNC_H
#define CONV_FUNC_H

#include <armadillo>

using namespace arma;
using namespace std;

/** @brief Make Tophat convolution
 *
 *  Generate Tophat convolution with an array input or scalar input.
 *
 * 	@param[in] half_base_width (double)
 * 	@param[in] radius_in_pix (mat)
 *
 *  @return Result convolution
 */
mat make_conv_func_tophat(const double half_base_width, const mat& radius_in_pix);

/** @brief Make Triangle convolution
 *
 *  Generate Triangle convolution with an array input or scalar input.
 *
 * 	@param[in] half_base_width (double)
 * 	@param[in] radius_in_pix (mat)
 * 	@param[in] triangle_value (double)
 *
 *  @return Result convolution
 */
mat make_conv_func_triangle(const double half_base_width, const mat& radius_in_pix, const double triangle_value);

/** @brief Make Sinc convolution
 *
 *  Generate Sinc convolution with an array input or scalar input.
 *
 * 	@param[in] radius_in_pix (mat)
 *
 *  @return Result convolution
 */
mat make_conv_func_sinc(const mat& radius_in_pix);

/** @brief Make Gaussian convolution
 *
 *  Generate Gaussian convolution with an array input or scalar input.
 *
 * 	@param[in] w1 (double)
 * 	@param[in] trunc (mat)
 *
 *  @return Result convolution
 */
mat make_conv_func_gaussian(const mat& trunc, const double w1);

/** @brief Make GaussianSinc convolution
 *
 *  Generate Gaussian Sinc convolution  with an array input or scalar input.
 *
 * 	@param[in] w1 (double)
 * 	@param[in] w2 (double)
 * 	@param[in] radius_in_pix (mat)
 *
 *  @return Result convolution
 */
mat make_conv_func_gaussian_sinc(const mat& radius_in_pix, const double w1, const double w2);

#endif /* CONV_FUNC_H */
