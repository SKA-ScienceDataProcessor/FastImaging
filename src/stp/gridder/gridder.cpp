/**
 * @file gridder.cpp
 * @brief Implementation of the gridder functions.
 */

#include "gridder.h"

namespace stp {

void convert_to_halfplane_visibilities(arma::mat& uv_in_pixels, arma::cx_mat& vis, arma::mat& vis_weights, int kernel_support)
{
    // Assume:  x = u = col 0
    //          y = v = col 1

    // Auxiliary arrays
    arma::mat uv_in_pixels_aux(arma::size(uv_in_pixels));
    arma::cx_mat vis_aux(arma::size(vis));
    arma::mat vis_weights_aux(arma::size(vis_weights));

    size_t i = 0;
    size_t j = 0;
    for (; i < uv_in_pixels.n_rows; ++i) {
        // Check if y value of the visibility point is negative (i.e. belongs to the top half-plane) and compute conjugate if true
        if (uv_in_pixels.at(i, 1) < 0.0) {
            // If the visibity point is close to the 0-frequency (within kernel_support distance)
            // also keep the visibility point in the top half-plane (use an auxiliary array)
            if (uv_in_pixels.at(i, 1) > -(kernel_support + 1)) {
                uv_in_pixels_aux.row(j) = uv_in_pixels.row(i);
                vis_aux.at(j) = vis.at(i);
                vis_weights_aux.at(j) = vis_weights.at(i);
                j++;
            }
            // Invert coordinates of the visibility point (change halfplane position)
            uv_in_pixels.row(i) *= (-1);
            // Also compute the conjugate of the visibility
            vis.at(i) = std::conj(vis.at(i));
        } else {
            // If the visibity point in the bottom halfplane is close to the 0-frequency (within kernel_support distance)
            // add the conjugate visibility point to the top half-plane (use an auxiliary array)
            if (uv_in_pixels.at(i, 1) < (kernel_support + 1)) {
                uv_in_pixels_aux.row(j) = uv_in_pixels.row(i) * (-1);
                vis_aux.at(j) = std::conj(vis.at(i));
                vis_weights_aux.at(j) = vis_weights.at(i);
                j++;
            }
        }
    }
    // Join the two arrays of visibilities and uv coordinates
    if (j) {
        uv_in_pixels = arma::join_cols(uv_in_pixels, uv_in_pixels_aux.rows(0, j - 1));
        vis = arma::join_cols(vis, vis_aux.rows(0, j - 1));
        vis_weights = arma::join_cols(vis_weights, vis_weights_aux.rows(0, j - 1));
    }
    assert(uv_in_pixels.n_rows == vis.n_rows);
    assert(uv_in_pixels.n_rows == vis_weights.n_rows);
    assert(uv_in_pixels.n_rows == (i + j));
}

arma::uvec bounds_check_kernel_centre_locations(arma::imat& kernel_centre_indices, int support, int image_size)
{
    arma::uvec good_vis(kernel_centre_indices.n_rows);

    int col = 0;
    kernel_centre_indices.each_row([&](arma::imat& r) {
        const int kc_x = r[0];
        const int kc_y = r[1];

        // Note that in-bound kernels that touch the left and top margins are also considered as being out-of-bounds (see comparison: <= 0).
        // This is to fix the non-symmetric issue on the complex gridded matrix caused by the even matrix size.
        if ((kc_x - support) <= 0 || (kc_y - support) <= 0 || (kc_x + support) >= image_size || (kc_y + support) >= image_size) {
            good_vis[col] = 0;
        } else {
            good_vis[col] = 1;
        }
        col++;
    });

    return std::move(good_vis);
}

arma::imat calculate_oversampled_kernel_indices(arma::mat& subpixel_coord, int oversampling)
{
    assert(arma::uvec(arma::find(arma::abs(subpixel_coord) > 0.5)).is_empty());

    arma::imat oversampled_coord(arma::size(subpixel_coord));

    int range_max = oversampling / 2;
    int range_min = -1 * range_max;

    for (arma::uword i = 0; i < subpixel_coord.n_elem; i++) {
        int val = rint(subpixel_coord.at(i) * oversampling);
        if (val > range_max) {
            val = range_max;
        } else {
            if (val < range_min) {
                val = range_min;
            }
        }
        oversampled_coord.at(i) = val;
    }

    return std::move(oversampled_coord);
}
}
