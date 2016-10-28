/** @file gridder.h
 *  @brief Functions of gridder
 *
 *  This contains the prototypes and implementation
 *  for the gridder functions
 *
 *  @bug No known bugs.
 */

#ifndef GRIDDER_FUNC_H
#define GRIDDER_FUNC_H

#include "../convolution/conv_func.h"
#include <float.h>

const int VIS_GRID_INDEX(0);
const int SAMPLING_GRID_INDEX(1);

/** @brief bounds_check_kernel_centre_locations function
 *
 *  Vectorized bounds check.
 *
 *  @param[in] uv (arma::mat): Array of uv co-ordinates
 *  @param[in] kernel_centre_indices(arma::mat): Corresponding array of
 *              nearest-pixel grid-locations, which will be the centre position
 *              of a kernel placement.
 *  @param[in] support (int): Kernel support size in regular pixels.
 *  @param[in] image_size (int): Image width in pixels
 *  @param[in] raise_if_bad (bool): If true, throw a ValueError if any bad locations
 *              are found, otherwise just log a warning message.
 *
 *  @return Index of good positions in the uv array.
 */
arma::uvec bounds_check_kernel_centre_locations(arma::mat uv, arma::mat kernel_centre_indices, int support, int image_size, bool raise_if_bad);

/** @brief convolve_to_grid function
 *
 *  Generate Tophat convolution with an array input or scalar input.
 *
 *  @param[in] support (int): Defines the 'radius' of the bounding box within
 *              which convolution takes place. `Box width in pixels = 2*support+1`.
 *              (The central pixel is the one nearest to the UV co-ordinates.)
 *              (This is sometimes known as the 'half-support')
 *  @param[in] image_size (int): Width of the image in pixels. NB we assume
 *              the pixel `[image_size//2,image_size//2]` corresponds to the origin
 *              in UV-space.
 *  @param[in] uv (arma::mat): UV-coordinates of visibilities.
 *              2d array of `float_`, shape: `(n_vis, 2)`.
 *              assumed ordering is u-then-v, i.e. `u, v = uv[idx]`
 *  @param[in] vis (arma::cx_mat): Complex visibilities.
 *              1d array, shape: `(n_vis,)`.
 *  @param[in] oversampling (double). Controls kernel-generation, see function
 *              description for details.
 *  @param[in] pad (bool).
 *  @param[in] raise_bounds (bool): Raise an exception if any of the UV
 *              samples lie outside (or too close to the edge) of the grid.
 *
 *  @return (arma::cx_cube) with 2 slices: vis_grid and sampling_grid; representing
 *            the gridded visibilities and the sampling weights.
 */
template <typename T, typename... Args>
arma::cx_cube convolve_to_grid(const int support, int image_size, arma::mat uv, arma::cx_mat vis, double oversampling, bool pad, bool normalize, bool raise_bounds, Args&&... args) {

    arma::mat uv_rounded = uv;
    uv_rounded.for_each([](arma::mat::elem_type& val){
        val = rint(val);
    });

    arma::mat uv_frac = uv - uv_rounded;
    arma::mat uv_rounded_int = arma::conv_to< arma::mat >::from(uv_rounded);
    double image_size_int = image_size / 2;
    arma::mat kernel_centre_on_grid = uv_rounded_int + repmat(arma::mat{image_size_int,image_size_int}, uv_rounded_int.n_rows, 1);
    arma::uvec good_vis_idx = bounds_check_kernel_centre_locations(uv, kernel_centre_on_grid, support, image_size, raise_bounds);

    arma::cx_mat vis_grid = arma::zeros<arma::cx_mat>(image_size, image_size);
    arma::cx_mat sampling_grid = vis_grid;
    arma::cx_mat typed_one = {1};
    arma::cx_cube result_grid(vis_grid.n_cols, vis_grid.n_rows, 2);

    int gc_x;
    int gc_y;
    std::complex<double> vis_value;

    for (int idx(0); idx < good_vis_idx.n_elem; ++idx)
    {
        vis_value = vis[good_vis_idx[idx]];
        gc_x = kernel_centre_on_grid(idx, 0);
        gc_y = kernel_centre_on_grid(idx, 1);

        arma::mat kernel = make_kernel_array<T>(support, uv_frac.row(idx), oversampling, pad, normalize, std::forward<Args>(args)...);
        arma::mat normed_kernel_array = kernel / accu(kernel);

        arma::span xrange = arma::span(gc_x - support, gc_x + support);
        arma::span yrange = arma::span(gc_y - support, gc_y + support );

        vis_grid(yrange, xrange) += vis_value * normed_kernel_array;
        sampling_grid(yrange, xrange) += typed_one[0] * normed_kernel_array;
    }

    result_grid.slice(VIS_GRID_INDEX) = vis_grid;
    result_grid.slice(SAMPLING_GRID_INDEX) = sampling_grid;

    return result_grid;
}

#endif /* GRIDDER_FUNC_H */
