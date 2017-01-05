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

#include <assert.h>
#include <cfloat>
#include <fstream>
#include <map>

#include "../convolution/conv_func.h"

namespace stp {

const int vis_grid_index = 0;
const int sampling_grid_index = 1;

/** @brief populate_kernel_cache function
 *
 *  Generate a cache of normalised kernels at oversampled-pixel offsets.
 *
 *  @param[in] T& kernel_creator : the kernel creator functor.
 *  @param[in] support (int): See kernel generation routine.
 *  @param[in] oversampling_val (int): Oversampling ratio value.
 *  @param[in] pad (bool) :  Whether to pad the array by an extra pixel-width.
 *             This is used when generating an oversampled kernel that will be used for interpolation. Default is false.
 *  @param[in] normalize (bool). Whether to normalize generated kernel functions. Default is true.
 *
 *  @return cache (std::map): Mapping oversampling-pixel offsets to normalised kernels.
 */
template <typename T>
std::map<std::pair<int, int>, arma::mat> populate_kernel_cache(const T& kernel_creator, const int support, const int oversampling_val, const bool pad = false, const bool normalize = true)
{
    int oversampled_pixel = (oversampling_val / 2);
    int cache_size = (oversampled_pixel * 2 + 1);

    arma::mat oversampled_pixel_offsets = arma::linspace(0, cache_size - 1, cache_size) - oversampled_pixel;
    std::map<std::pair<int, int>, arma::mat> cache;

    oversampled_pixel_offsets.for_each([&oversampled_pixel_offsets, &oversampling_val, &support, &pad, &normalize, &kernel_creator, &cache](arma::mat::elem_type& val_x) {
        oversampled_pixel_offsets.for_each([&val_x, &oversampled_pixel_offsets, &oversampling_val, &support, &pad, &normalize, &kernel_creator, &cache](arma::mat::elem_type& val_y) {
            arma::mat subpixel_offset = arma::mat({ val_x, val_y }) / oversampling_val;
            arma::mat kernel = make_kernel_array(kernel_creator, support, subpixel_offset, std::experimental::nullopt, pad, normalize);
            cache[std::make_pair(val_x, val_y)] = kernel;
        });
    });

    return cache;
}

/** @brief bounds_check_kernel_centre_locations function
 *
 *  Vectorized bounds check.
 *
 *  @param[in] kernel_centre_indices(arma::mat): Corresponding array of
 *              nearest-pixel grid-locations, which will be the centre position
 *              of a kernel placement.
 *  @param[in] support (int): Kernel support size in regular pixels.
 *  @param[in] image_size (int): Image width in pixels
 *
 *  @return Index of good positions in the uv array.
 */
arma::uvec bounds_check_kernel_centre_locations(arma::mat kernel_centre_indices, int support, int image_size);

/** @brief calculate_oversampled_kernel_indices function
 *
 *  Find the nearest oversampled gridpoint for given sub-pixel offset.
 *
 *  @param[in] subpixel_coord (arma::mat): Array of 'fractional' co-ords, that is the
               subpixel offsets from nearest pixel on the regular grid.
 *  @param[in] oversampling_val (int). How many oversampled pixels to one regular pixel.
 *
 *  @return oversampled_kernel_idx (arma::mat): Corresponding oversampled pixel
        indexes
 */
arma::mat calculate_oversampled_kernel_indices(arma::mat subpixel_coord, const int oversampling_val);

/** @brief convolve_to_grid function
 *
 *  Grid visibilities, calculating the exact kernel distribution for each.
 *
 *  @param[in] T& kernel_creator : the kernel creator functor.
 *  @param[in] support (int) : Defines the 'radius' of the bounding box within
 *              which convolution takes place. `Box width in pixels = 2*support+1`.
 *              (The central pixel is the one nearest to the UV co-ordinates.)
 *              (This is sometimes known as the 'half-support')
 *  @param[in] image_size (int) : Width of the image in pixels. NB we assume
 *              the pixel `[image_size//2,image_size//2]` corresponds to the origin
 *              in UV-space.
 *  @param[in] uv (arma::mat) : UV-coordinates of visibilities.
 *  @param[in] vis (arma::cx_mat): Complex visibilities. 1d array, shape: `(n_vis,)`.
 *  @param[in] oversampling (optional<int>): Controls kernel-generation
 *  @param[in] pad (bool) :  Whether to pad the array by an extra pixel-width.
 *             This is used when generating an oversampled kernel that will be used for interpolation. Default is false.
 *  @param[in] normalize (bool). Whether to normalize generated kernel functions. Default is true.
 *
 *  @return (std::pair<arma::cx_mat, arma::cx_mat>) 2 slices: vis_grid and sampling_grid; representing
 *            the gridded visibilities and the sampling weights.
 */
template <typename T>
std::pair<arma::cx_mat, arma::cx_mat> convolve_to_grid(const T& kernel_creator, const int support, int image_size, arma::mat uv, arma::cx_mat vis, const std::experimental::optional<int>& oversampling = std::experimental::nullopt, bool pad = false, bool normalize = true)
{
    assert(uv.n_cols == 2);
    assert(uv.n_rows == vis.n_rows);

    arma::mat uv_rounded_int = uv;
    uv_rounded_int.for_each([](arma::mat::elem_type& val) {
        val = rint(val);
    });

    arma::mat uv_frac = uv - uv_rounded_int;
    double image_size_int = image_size / 2;
    arma::mat kernel_centre_on_grid = uv_rounded_int + image_size_int;
    arma::uvec good_vis_idx = bounds_check_kernel_centre_locations(kernel_centre_on_grid, support, image_size);

    arma::cx_mat vis_grid = arma::zeros<arma::cx_mat>(image_size, image_size);
    arma::cx_mat sampling_grid = vis_grid;

    std::map<std::pair<int, int>, arma::mat> kernel_cache;
    arma::mat oversampled_offset;

    if (bool(oversampling) == true) {
        // If an integer value is supplied (oversampling), we pre-generate an oversampled kernel ahead of time.
        kernel_cache = populate_kernel_cache(kernel_creator, support, (*oversampling), pad, normalize);
        oversampled_offset = calculate_oversampled_kernel_indices(uv_frac, (*oversampling));
    }

    arma::uword idx(0);

    good_vis_idx.for_each([&idx, &kernel_centre_on_grid, &support, &oversampling, &kernel_cache, &oversampled_offset, &uv_frac, &pad, &normalize, &kernel_creator, &vis, &vis_grid, &sampling_grid](arma::uvec::elem_type& val) {
        arma::uword gc_x = kernel_centre_on_grid(idx, 0);
        arma::uword gc_y = kernel_centre_on_grid(idx, 1);

        arma::span xrange = arma::span(gc_x - support, gc_x + support);
        arma::span yrange = arma::span(gc_y - support, gc_y + support);

        arma::mat normed_kernel_array;

        if (bool(oversampling) == true) {
            // We pick the pre-generated kernel corresponding to the sub-pixel offset nearest to that of the visibility.
            normed_kernel_array = kernel_cache[std::make_pair(oversampled_offset.at(idx, 0), oversampled_offset.at(idx, 1))];
        } else {
            // Exact gridding is used, i.e. the kernel is recalculated for each visibility, with
            // precise sub-pixel offset according to that visibility's UV co-ordinates.
            normed_kernel_array = make_kernel_array(kernel_creator, support, uv_frac.row(idx), oversampling, pad, normalize);
        }

        vis_grid(yrange, xrange) += vis[val] * normed_kernel_array;
        sampling_grid(yrange, xrange) += std::complex<double>(1) * normed_kernel_array;
        idx++;
    });

    return std::make_pair(std::move(vis_grid), std::move(sampling_grid));
}
}

#endif /* GRIDDER_FUNC_H */
