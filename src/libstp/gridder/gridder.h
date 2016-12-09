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

#include <cfloat>
#include <fstream>
#include <map>

#include "../convolution/conv_func.h"

const int VIS_GRID_INDEX(0);
const int SAMPLING_GRID_INDEX(1);

/** @brief populate_kernel_cache function
 *
 *  Generate a cache of normalised kernels at oversampled-pixel offsets.
 *
 *  @param[in] support (int): See kernel generation routine.
 *  @param[in] oversampling (float): Oversampling ratio.
 *  @param[in] pad (bool).
 *  @param[in] normalize (bool).
 *  @param[in] T& kernel_creator : the kernel creator functor
 *
 *  @return cache (std::map): Mapping oversampling-pixel offsets to normalised kernels.
 */
template <typename T>
std::map<std::pair<int, int>, arma::mat> populate_kernel_cache(const int support, const float oversampling, bool pad, bool normalize, const T& kernel_creator)
{

    int oversampled_pixel(oversampling / 2);
    int cache_size(oversampled_pixel * 2 + 1);

    arma::mat oversampled_pixel_offsets = arma::linspace(0, cache_size - 1, cache_size) - oversampled_pixel;
    std::map<std::pair<int, int>, arma::mat> cache;

    oversampled_pixel_offsets.for_each([&oversampled_pixel_offsets, &oversampling, &support, &pad, &normalize, &kernel_creator, &cache](arma::mat::elem_type& val_x) {
        oversampled_pixel_offsets.for_each([&val_x, &oversampled_pixel_offsets, &oversampling, &support, &pad, &normalize, &kernel_creator, &cache](arma::mat::elem_type& val_y) {
            arma::mat subpixel_offset = arma::mat({ val_x, val_y }) / oversampling;
            arma::mat kernel = make_kernel_array(support, subpixel_offset, oversampling_disabled, pad, normalize, kernel_creator);
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
 *  @param[in] oversampling (double). How many oversampled pixels to one regular pixel.
 *
 *  @return oversampled_kernel_idx (arma::mat): Corresponding oversampled pixel
        indexes
 */
arma::mat calculate_oversampled_kernel_indices(arma::mat subpixel_coord, const double oversampling);

/** @brief convolve_to_grid function
 *
 *  Grid visibilities, calculating the exact kernel distribution for each.
 *
 *  @param[in] support (int) : Defines the 'radius' of the bounding box within
 *              which convolution takes place. `Box width in pixels = 2*support+1`.
 *              (The central pixel is the one nearest to the UV co-ordinates.)
 *              (This is sometimes known as the 'half-support')
 *  @param[in] image_size (int) : Width of the image in pixels. NB we assume
 *              the pixel `[image_size//2,image_size//2]` corresponds to the origin
 *              in UV-space.
 *  @param[in] uv (arma::mat) : UV-coordinates of visibilities.
 *  @param[in] vis (arma::cx_mat): Complex visibilities. 1d array, shape: `(n_vis,)`.
 *  @param[in] oversampling (double): Controls kernel-generation
 *  @param[in] pad (bool) :  Whether to pad the array by an extra pixel-width.
 *             This is used when generating an oversampled kernel that will be used for interpolation.
 *  @param[in] normalize (bool).
 *
 *  @return (arma::cx_cube) with 2 slices: vis_grid and sampling_grid; representing
 *            the gridded visibilities and the sampling weights.
 */
template <typename T>
arma::cx_cube convolve_to_grid(const int support, int image_size, arma::mat uv, arma::cx_mat vis, double oversampling, bool pad, bool normalize, const T& kernel_creator)
{

    arma::mat uv_rounded = uv;
    uv_rounded.for_each([](arma::mat::elem_type& val) {
        val = rint(val);
    });

    arma::mat uv_frac = uv - uv_rounded;
    arma::mat uv_rounded_int = arma::conv_to<arma::mat>::from(uv_rounded);
    double image_size_int = image_size / 2;
    arma::mat kernel_centre_on_grid = uv_rounded_int + repmat(arma::mat{ image_size_int, image_size_int }, uv_rounded_int.n_rows, 1);
    arma::uvec good_vis_idx = bounds_check_kernel_centre_locations(kernel_centre_on_grid, support, image_size);

    arma::cx_mat vis_grid = arma::zeros<arma::cx_mat>(image_size, image_size);
    arma::cx_mat sampling_grid = vis_grid;
    arma::cx_mat typed_one = { std::complex<double>(1) };
    arma::cx_cube result_grid(vis_grid.n_cols, vis_grid.n_rows, 2);

    std::map<std::pair<int, int>, arma::mat> kernel_cache;
    arma::mat oversampled_offset;

    if (oversampling < oversampling_disabled) {
        // If an integer value is supplied (oversampling), we pre-generate an oversampled kernel ahead of time.
        kernel_cache = populate_kernel_cache(support, oversampling, pad, normalize, kernel_creator);
        oversampled_offset = calculate_oversampled_kernel_indices(uv_frac, oversampling);
    }

    arma::uword idx(0);

    good_vis_idx.for_each([&idx, &kernel_centre_on_grid, &support, &oversampling, &kernel_cache, &oversampled_offset, &uv_frac, &pad, &normalize, &kernel_creator, &vis, &vis_grid, &sampling_grid, &typed_one](arma::uvec::elem_type& val) {
        arma::uword gc_x = kernel_centre_on_grid(idx, 0);
        arma::uword gc_y = kernel_centre_on_grid(idx, 1);

        arma::span xrange = arma::span(gc_x - support, gc_x + support);
        arma::span yrange = arma::span(gc_y - support, gc_y + support);

        arma::mat normed_kernel_array;

        if (oversampling < oversampling_disabled) {
            // We pick the pre-generated kernel corresponding to the sub-pixel offset nearest to that of the visibility.
            normed_kernel_array = kernel_cache[std::make_pair(oversampled_offset.at(idx, 0), oversampled_offset.at(idx, 1))];
        } else {
            // Exact gridding is used, i.e. the kernel is recalculated for each visibility, with
            // precise sub-pixel offset according to that visibility's UV co-ordinates.
            arma::mat kernel = make_kernel_array(support, uv_frac.row(idx), oversampling, pad, normalize, kernel_creator);
            normed_kernel_array = kernel / accu(kernel);
        }

        vis_grid(yrange, xrange) += vis[val] * normed_kernel_array;
        sampling_grid(yrange, xrange) += typed_one[0] * normed_kernel_array;
        idx++;
    });

    result_grid.slice(VIS_GRID_INDEX) = vis_grid;
    result_grid.slice(SAMPLING_GRID_INDEX) = sampling_grid;

    return result_grid;
}

#endif /* GRIDDER_FUNC_H */
