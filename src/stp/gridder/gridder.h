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
#include <tbb/tbb.h>

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
 *  @param[in] oversampling (int): Oversampling ratio value.
 *  @param[in] pad (bool) :  Whether to pad the array by an extra pixel-width.
 *             This is used when generating an oversampled kernel that will be used for interpolation. Default is false.
 *  @param[in] normalize (bool). Whether to normalize generated kernel functions. Default is true.
 *
 *  @return cache (std::map): Mapping oversampling-pixel offsets to normalised kernels.
 */
template <typename T>
arma::field<arma::mat> populate_kernel_cache(const T& kernel_creator, const int support, const int oversampling, const bool pad = false, const bool normalize = true)
{
    int oversampled_pixel = (oversampling / 2);
    int cache_size = (oversampled_pixel * 2 + 1);

    arma::mat oversampled_pixel_offsets = (arma::linspace(0, cache_size - 1, cache_size) - oversampled_pixel) / oversampling;
    arma::field<arma::mat> cache(cache_size, cache_size); // 2D kernel array cache to be returned
    arma::field<arma::mat> kernel1D_cache(cache_size); // Temporary cache for 1D kernel arrays

    // Fill 1D kernel array cache
    for (int i = 0; i < cache_size; i++) {
        kernel1D_cache(i) = make_1D_kernel(kernel_creator, support, oversampled_pixel_offsets[i], 1, pad, normalize);
    }

    // Generate 2D kernel array cache based on 1D kernel cache for reduced running times
    for (int i = 0; i < cache_size; i++) {
        for (int j = 0; j <= i; j++) {
            arma::mat result = kernel1D_cache(j) * kernel1D_cache(i).st();
            cache(j, i) = result;
            if (i != j)
                cache(i, j) = result.st();
        }
    }

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
 *  @return (uvec): List of indices for 'good' (in-bounds) positions. Note this is a list of integer index values.
 */
arma::uvec bounds_check_kernel_centre_locations(arma::mat& kernel_centre_indices, int support, int image_size);

/** @brief calculate_oversampled_kernel_indices function
 *
 *  Find the nearest oversampled gridpoint for given sub-pixel offset.
 *
 *  @param[in] subpixel_coord (arma::mat): Array of 'fractional' co-ords, that is the
               subpixel offsets from nearest pixel on the regular grid.
 *  @param[in] oversampling (int). How many oversampled pixels to one regular pixel.
 *
 *  @return oversampled_kernel_idx (arma::mat): Corresponding oversampled pixel
        indexes
 */
arma::mat calculate_oversampled_kernel_indices(arma::mat subpixel_coord, int oversampling);

/** @brief convolve_to_grid function
 *
 * Grid visibilities, calculating the exact kernel distribution for each.
 *
 * If ``exact == True`` then exact gridding is used, i.e. the kernel is
 *  recalculated for each visibility, with precise sub-pixel offset according to
 *  that visibility's UV co-ordinates. Otherwise, instead of recalculating the
 *  kernel for each sub-pixel location, we pre-generate an oversampled kernel
 *  ahead of time - so e.g. for an oversampling of 5, the kernel is
 *  pre-generated at 0.2 pixel-width offsets. We then pick the pre-generated
 *  kernel corresponding to the sub-pixel offset nearest to that of the
 *  visibility.
 *  Kernel pre-generation results in improved performance, particularly with
 *  large numbers of visibilities and complex kernel functions, at the cost of
 *  introducing minor aliasing effects due to the 'step-like' nature of the
 *  oversampled kernel. This in turn can be minimised (at the cost of longer
 *  start-up times and larger memory usage) by pre-generating kernels with a
 *  larger oversampling ratio, to give finer interpolation.
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
 *  @param[in] kernel_exact (bool): Calculate exact kernel-values for every UV-sample.
 *  @param[in] oversampling (int): Controls kernel-generation if ``exact==False``.
            Larger values give a finer-sampled set of pre-cached kernels.
 *  @param[in] pad (bool) :  Whether to pad the array by an extra pixel-width.
 *             This is used when generating an oversampled kernel that will be used for interpolation. Default is false.
 *  @param[in] normalize (bool). Whether to normalize generated kernel functions. Default is true.
 *
 *  @return (std::pair<arma::cx_mat, arma::cx_mat>) 2 slices: vis_grid and sampling_grid; representing
 *            the gridded visibilities and the sampling weights.
 */
template <typename T>
std::pair<arma::cx_mat, arma::cx_mat> convolve_to_grid(const T& kernel_creator, const int support, int image_size, arma::mat uv, arma::cx_mat vis, bool kernel_exact = true, int oversampling = 1, bool pad = false, bool normalize = true)
{
    assert(uv.n_cols == 2);
    assert(uv.n_rows == vis.n_rows);
    assert(kernel_exact || (oversampling >= 1));

    if (kernel_exact == true)
        oversampling = 1;

    arma::mat uv_rounded = uv;
    uv_rounded.for_each([](arma::mat::elem_type& val) {
        val = rint(val);
    });

    arma::mat uv_frac = uv - uv_rounded;
    double image_size_int = image_size / 2;
    arma::mat kernel_centre_on_grid = uv_rounded + image_size_int;
    arma::uvec good_vis_idx = bounds_check_kernel_centre_locations(kernel_centre_on_grid, support, image_size);

    arma::cx_mat vis_grid = arma::zeros<arma::cx_mat>(image_size, image_size);
    arma::mat sampling_grid = arma::zeros<arma::mat>(image_size, image_size);
    int kernel_size = support * 2 + 1;

    if (kernel_exact == false) {
        // If an integer value is supplied (oversampling), we pre-generate an oversampled kernel ahead of time.
        arma::field<arma::mat> kernel_cache = populate_kernel_cache(kernel_creator, support, oversampling, pad, normalize);
        arma::mat oversampled_offset = calculate_oversampled_kernel_indices(uv_frac, oversampling);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, kernel_size, 1), [&good_vis_idx, &kernel_centre_on_grid, &support, &kernel_size, &oversampling, &kernel_cache, &oversampled_offset, &uv_frac, &kernel_creator, &vis, &vis_grid, &sampling_grid](const tbb::blocked_range<size_t>& r) {

            good_vis_idx.for_each([&kernel_centre_on_grid, &support, &kernel_size, &oversampling, &kernel_cache, &oversampled_offset, &uv_frac, &kernel_creator, &vis, &vis_grid, &sampling_grid, &r](arma::uvec::elem_type& val) {
                arma::uword gc_x = kernel_centre_on_grid(val, 0);
                arma::uword gc_y = kernel_centre_on_grid(val, 1);
                arma::uword cp_x = oversampled_offset.at(val, 0) + (oversampling / 2);
                arma::uword cp_y = oversampled_offset.at(val, 1) + (oversampling / 2);

                int conv_col = ((gc_x - r.begin() + kernel_size) % kernel_size);
                if (conv_col > support)
                    conv_col -= kernel_size;
                int grid_col = (gc_x - conv_col);

                assert(abs(conv_col) <= support);
                assert((grid_col % kernel_size) == (int)r.begin());

                //arma::span yrange = arma::span(gc_y - support, gc_y + support);

                // We pick the pre-generated kernel corresponding to the sub-pixel offset nearest to that of the visibility.
                //vis_grid(yrange, grid_col) += vis[val] * kernel_cache(cp_y, cp_x).col(support - conv_col);
                //sampling_grid(yrange, grid_col) += kernel_cache(cp_y, cp_x).col(support - conv_col);
                for (int i = 0; i < kernel_size; i++) {
                    vis_grid.at(gc_y - support + i, grid_col) += vis[val] * kernel_cache(cp_y, cp_x).at(i, support - conv_col);
                    sampling_grid.at(gc_y - support + i, grid_col) += kernel_cache(cp_y, cp_x).at(i, support - conv_col);
                }
            });
        });
    } else {
        good_vis_idx.for_each([&kernel_centre_on_grid, &support, &uv_frac, &pad, &normalize, &kernel_creator, &vis, &vis_grid, &sampling_grid](arma::uvec::elem_type& val) {
            arma::uword gc_x = kernel_centre_on_grid(val, 0);
            arma::uword gc_y = kernel_centre_on_grid(val, 1);

            arma::span xrange = arma::span(gc_x - support, gc_x + support);
            arma::span yrange = arma::span(gc_y - support, gc_y + support);

            // Exact gridding is used, i.e. the kernel is recalculated for each visibility, with
            // precise sub-pixel offset according to that visibility's UV co-ordinates.
            arma::mat normed_kernel_array = make_kernel_array(kernel_creator, support, uv_frac.row(val));
            vis_grid(yrange, xrange) += vis[val] * normed_kernel_array;
            sampling_grid(yrange, xrange) += normed_kernel_array;
        });
    }

    return std::make_pair(std::move(vis_grid), std::move(arma::conv_to<arma::cx_mat>::from(sampling_grid)));
}
}

#endif /* GRIDDER_FUNC_H */
