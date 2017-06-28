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

#include <cassert>
#include <cfloat>
#include <fstream>
#include <map>
#include <tbb/tbb.h>

#include "../common/matstp.h"
#include "../convolution/conv_func.h"
#include "../types.h"

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
 *  @param[in] kernel_centre_indices(arma::imat): Corresponding array of
 *              nearest-pixel grid-locations, which will be the centre position
 *              of a kernel placement.
 *  @param[in] support (int): Kernel support size in regular pixels.
 *  @param[in] image_size (int): Image width in pixels
 *
 *  @return (uvec): List of indices for 'good' (in-bounds) positions. Note this is a list of integer index values.
 */
arma::uvec bounds_check_kernel_centre_locations(arma::imat& kernel_centre_indices, int support, int image_size);

/** @brief calculate_oversampled_kernel_indices function
 *
 *  Find the nearest oversampled gridpoint for given sub-pixel offset.
 *
 *  @param[in] subpixel_coord (arma::mat): Array of 'fractional' co-ords, that is the
               subpixel offsets from nearest pixel on the regular grid.
 *  @param[in] oversampling (int). How many oversampled pixels to one regular pixel.
 *
 *  @return oversampled_kernel_idx (arma::imat): Corresponding oversampled pixel indexes
 */
arma::imat calculate_oversampled_kernel_indices(arma::mat& subpixel_coord, int oversampling);

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
 *  @return (std::pair<arma::cx_mat, arma::mat>) 2 slices: vis_grid and sampling_grid; representing
 *            the gridded visibilities and the sampling weights.
 */
template <typename T>
std::pair<MatStp<cx_real_t>, MatStp<cx_real_t>> convolve_to_grid(const T& kernel_creator, const int support, int image_size, const arma::mat& uv, const arma::cx_mat& vis,
    bool kernel_exact = true, int oversampling = 1, bool pad = false, bool normalize = true, bool shift_uv = true, bool halfplane_gridding = true)
{
    assert(uv.n_cols == 2);
    assert(uv.n_rows == vis.n_rows);
    assert(kernel_exact || (oversampling >= 1));
    assert((image_size % 2) == 0);

    if (kernel_exact == true)
        oversampling = 1;

    arma::mat uv_rounded = uv;
    uv_rounded.for_each([](arma::mat::elem_type& val) {
        val = rint(val);
    });

    arma::mat uv_frac = uv - uv_rounded;
    arma::imat kernel_centre_on_grid = arma::conv_to<arma::imat>::from(uv_rounded) + (image_size / 2);

    // Check bounds
    arma::uvec good_vis_idx = bounds_check_kernel_centre_locations(kernel_centre_on_grid, support, image_size);

    // Number of rows of the output images. This changes if halfplane gridding is used
    int image_rows = image_size;
    if (halfplane_gridding) {
        image_rows = (image_size / 2) + 1;
    }

    // Shift positions of the visibilities (to avoid call to fftshift function)
    if (shift_uv) {
        int shift_offset = image_size / 2;
        kernel_centre_on_grid.each_row([&](arma::imat& r) {
            if (r[0] < shift_offset) {
                r[0] += shift_offset;
            } else {
                r[0] -= shift_offset;
            }
            if (r[1] < shift_offset) {
                r[1] += shift_offset;
            } else {
                r[1] -= shift_offset;
            }
        });
    }

    // Create matrices for output gridded images
    // Use MatStp class because these images shall be efficiently initialized with zeros
    MatStp<cx_real_t> vis_grid(image_rows, image_size);
    MatStp<cx_real_t> sampling_grid(image_rows, image_size);

    int kernel_size = support * 2 + 1;

    if (kernel_exact == false) {
        // Kernel oversampled gridder
        // If an integer value is supplied (oversampling), we pre-generate an oversampled kernel ahead of time.
        const arma::field<arma::mat> kernel_cache = populate_kernel_cache(kernel_creator, support, oversampling, pad, normalize);
        arma::imat oversampled_offset = calculate_oversampled_kernel_indices(uv_frac, oversampling) + (oversampling / 2);

        /*
        // Single-threaded implementation of oversampled gridder
        good_vis_idx.for_each([&](arma::uvec::elem_type& val) {
            const int gc_x = kernel_centre_on_grid(val, 0);
            const int gc_y = kernel_centre_on_grid(val, 1);

            const arma::uword cp_x = oversampled_offset.at(val, 0);
            const arma::uword cp_y = oversampled_offset.at(val, 1);

            for (int j = 0; j < kernel_size; j++) {
                int grid_col = gc_x - support + j;
                if (grid_col < 0) // left/right split of the kernel
                    grid_col += image_size;
                if (grid_col >= image_size) // left/right split of the kernel
                    grid_col -= image_size;

                for (int i = 0; i < kernel_size; i++) {
                    int grid_row = gc_y - support + i;
                    if (grid_row < 0) // top/bottom split of the kernel. Halfplane gridding: kernel points in the negative halfplane are excluded
                        grid_row += image_size;
                    if (grid_row >= image_size) // top/bottom split of the kernel. Halfplane gridding: kernel points touching the positive halfplane are used
                        grid_row -= image_size;
                    // The following condition is needed for the case of halfplane gridding, because only the top halfplane visibilities are convolved
                    const double kernel_val = kernel_cache(cp_y, cp_x).at(i, j);
                    if (grid_row < image_rows) {
                        vis_grid.at(grid_row, grid_col) += vis[val] * kernel_val;
                        sampling_grid.at(grid_row, grid_col) += kernel_val;
                    }
                }
            }
        });*/

        // Multi-threaded implementation of oversampled gridder (20-25% faster)
        tbb::parallel_for(tbb::blocked_range<size_t>(0, kernel_size, 1), [&](const tbb::blocked_range<size_t>& r) {

            for (arma::uword k = 0; k < good_vis_idx.n_elem; k++) {
                arma::uword val = good_vis_idx.at(k);
                int gc_x = kernel_centre_on_grid(val, 0);
                int gc_y = kernel_centre_on_grid(val, 1);

                const arma::uword cp_x = oversampled_offset.at(val, 0);
                const arma::uword cp_y = oversampled_offset.at(val, 1);

                int conv_col = ((gc_x - r.begin() + kernel_size) % kernel_size);
                if (conv_col > support)
                    conv_col -= kernel_size;
                int grid_col = (gc_x - conv_col);
                if ((grid_col < image_size) && (grid_col >= 0)) {

                    assert(std::abs(conv_col) <= support);
                    assert(arma::uword(grid_col % kernel_size) == r.begin());

                    // Pick the pre-generated kernel corresponding to the sub-pixel offset nearest to that of the visibility.
                    for (int i = 0; i < kernel_size; ++i) {
                        const double kernel_val = kernel_cache(cp_y, cp_x).at(i, support - conv_col);
                        int grid_row = gc_y - support + i;
                        if (grid_row < 0)
                            grid_row += image_size;
                        if (grid_row >= image_size)
                            grid_row -= image_size;
                        if (grid_row < image_rows) {
                            vis_grid.at(grid_row, grid_col) += vis[val] * kernel_val;
                            sampling_grid.at(grid_row, grid_col) += kernel_val;
                        }
                    }
                }

                assert(gc_x < image_size);

                // This is for the cases when the kernel is split between the left and right margins
                if ((gc_x >= (image_size - support)) || (gc_x < support)) {
                    if (gc_x >= (image_size - support)) {
                        gc_x -= image_size;
                        conv_col = ((gc_x - r.begin() + kernel_size * 2) % kernel_size);
                        if (conv_col > support)
                            conv_col -= kernel_size;
                        grid_col = (gc_x - conv_col);
                        assert(std::abs(conv_col) <= support);
                    } else {
                        assert(gc_x < support);
                        gc_x += image_size;
                        conv_col = ((gc_x - r.begin() + kernel_size) % kernel_size);
                        if (conv_col > support)
                            conv_col -= kernel_size;
                        grid_col = (gc_x - conv_col);
                        assert(std::abs(conv_col) <= support);
                    }

                    if ((grid_col < image_size) && (grid_col >= 0)) {
                        // Pick the pre-generated kernel corresponding to the sub-pixel offset nearest to that of the visibility.
                        for (int i = 0; i < kernel_size; i++) {
                            const double kernel_val = kernel_cache(cp_y, cp_x).at(i, support - conv_col);
                            int grid_row = gc_y - support + i;
                            if (grid_row < 0)
                                grid_row += image_size;
                            if (grid_row >= image_size)
                                grid_row -= image_size;
                            if (grid_row < image_rows) {
                                vis_grid.at(grid_row, grid_col) += vis[val] * kernel_val;
                                sampling_grid.at(grid_row, grid_col) += kernel_val;
                            }
                        }
                    }
                }
            }
        });

    } else {
        // Exact gridder (slower but with more accuracy)
        good_vis_idx.for_each([&](arma::uvec::elem_type& val) {
            const int gc_x = kernel_centre_on_grid(val, 0);
            const int gc_y = kernel_centre_on_grid(val, 1);

            // Exact gridding is used, i.e. the kernel is recalculated for each visibility, with
            // precise sub-pixel offset according to that visibility's UV co-ordinates.
            arma::mat normed_kernel_array = make_kernel_array(kernel_creator, support, uv_frac.row(val));

            for (int j = 0; j < kernel_size; j++) {
                int grid_col = gc_x - support + j;
                if (grid_col < 0) // left/right split of the kernel
                    grid_col += image_size;
                if (grid_col >= image_size) // left/right split of the kernel
                    grid_col -= image_size;

                for (int i = 0; i < kernel_size; i++) {
                    int grid_row = gc_y - support + i;
                    if (grid_row < 0) // top/bottom split of the kernel. Halfplane gridding: kernel points in the negative halfplane are excluded
                        grid_row += image_size;
                    if (grid_row >= image_size) // top/bottom split of the kernel. Halfplane gridding: kernel points touching the positive halfplane are used
                        grid_row -= image_size;
                    // The following condition is needed for the case of halfplane gridding, because only the top halfplane visibilities are convolved
                    if (grid_row < image_rows) {
                        vis_grid(grid_row, grid_col) += vis[val] * normed_kernel_array.at(i, j);
                        sampling_grid(grid_row, grid_col) += normed_kernel_array.at(i, j);
                    }
                }
            }
        });
    }

    return std::make_pair(std::move(vis_grid), std::move(sampling_grid));
}
}

#endif /* GRIDDER_FUNC_H */
