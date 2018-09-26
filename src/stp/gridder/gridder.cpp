/**
 * @file gridder.cpp
 * @brief Implementation of the gridder functions.
 */

#include "gridder.h"

namespace stp {

void convert_to_halfplane_visibilities(arma::mat& uv_lambda, arma::cx_mat& vis, int kernel_support, arma::uvec& good_vis)
{
    // Assume:  x = u = col 0
    //          y = v = col 1
    size_t n_rows = uv_lambda.n_rows;

    for (size_t i = 0; i < n_rows; ++i) {
        // Check if y value of the visibility point is negative (i.e. belongs to the top half-plane) and compute conjugate if true
        if (uv_lambda.at(i, 1) < 0.0) {
            // If the visibity point is close to the 0-frequency (within kernel_support distance)
            // also keep the visibility point in the top half-plane
            if (uv_lambda.at(i, 1) > -(kernel_support + 1)) {
                good_vis.at(i) = 2;
            }
            // Invert coordinates of the visibility point (change halfplane position)
            uv_lambda.row(i) *= (-1);
            // Also compute the conjugate of the visibility
            vis.at(i) = std::conj(vis.at(i));
        } else {
            // If the visibity point in the bottom halfplane is close to the 0-frequency (within kernel_support distance)
            // add the conjugate visibility point to the top half-plane
            if (uv_lambda.at(i, 1) < (kernel_support + 1)) {
                good_vis.at(i) = 2;
            }
        }
    }
}

void convert_to_halfplane_visibilities(arma::mat& uv_lambda, arma::vec& w_lambda, arma::cx_mat& vis, int kernel_support, arma::uvec& good_vis)
{
    // Assume:  x = u = col 0
    //          y = v = col 1
    size_t n_rows = uv_lambda.n_rows;

    for (size_t i = 0; i < n_rows; ++i) {
        // Check if y value of the visibility point is negative (i.e. belongs to the top half-plane) and compute conjugate if true
        if (uv_lambda.at(i, 1) < 0.0) {
            // If the visibity point is close to the 0-frequency (within kernel_support distance)
            // also keep the visibility point in the top half-plane
            if (uv_lambda.at(i, 1) > -(kernel_support + 1)) {
                good_vis.at(i) = 2;
            }
            // Invert coordinates of the visibility point (change halfplane position)
            uv_lambda.row(i) *= (-1);
            w_lambda(i) *= (-1);
            // Also compute the conjugate of the visibility
            vis.at(i) = std::conj(vis.at(i));
        } else {
            // If the visibity point in the bottom halfplane is close to the 0-frequency (within kernel_support distance)
            // add the conjugate visibility point to the top half-plane
            if (uv_lambda.at(i, 1) < (kernel_support + 1)) {
                good_vis.at(i) = 2;
            }
        }
    }
}

void average_w_planes(arma::mat w_lambda, const arma::uvec& good_vis, int num_wplanes, arma::vec& w_avg_values, arma::ivec& w_planes_firstidx, bool median)
{
    int num_elems = good_vis.n_elem;
    int num_gvis = 0;
    // Count good visibilities
    for (size_t i = 0; i < good_vis.n_elem; ++i) {
        if (good_vis[i] != 0) {
            num_gvis++;
        }
    }
    int plane_size = std::ceil(double(num_gvis) / num_wplanes);
    assert(plane_size >= 1);

    if (median) {

        int begin = 0;
        for (int idx = 0; idx < num_wplanes; idx++) {

            int idx_mid = 0;
            int middle = plane_size / 2;
            if ((begin + plane_size) > num_gvis) {
                middle = (num_gvis - begin) / 2;
            }

            // average w-plane components
            int k;
            int counter = 0;
            for (k = begin; (counter < plane_size) && (k < num_elems); k++) {
                if (good_vis(k) != 0) {
                    if (counter == middle) {
                        idx_mid = k;
                    }
                    counter++;
                }
            }
            if ((plane_size % 2) == 0) {
                // Even number
                w_avg_values[idx] = (w_lambda(idx_mid - 1) + w_lambda(idx_mid)) / 2.0;
            } else {
                // Odd number
                w_avg_values[idx] = w_lambda(idx_mid);
            }

            w_planes_firstidx[idx] = begin;
            begin = k;
        }

    } else {
        int begin = 0;
        for (int idx = 0; idx < num_wplanes; idx++) {

            // average w-plane components
            double sum = 0;
            int k;
            int counter = 0;
            for (k = begin; (counter < plane_size) && (k < num_elems); k++) {
                if (good_vis(k) != 0) {
                    sum += w_lambda(k);
                    counter++;
                }
            }
            w_planes_firstidx[idx] = begin;
            begin = k;
            if (counter > 0)
                w_avg_values[idx] = sum / counter;
        }
    }
}

void bounds_check_kernel_centre_locations(arma::uvec& good_vis, const arma::imat& kernel_centre_on_grid, int image_size, int support)
{
    // Check bounds
    int col = 0;
    kernel_centre_on_grid.each_row([&](const arma::imat& r) {
        const auto& kc_x = r[0];
        const auto& kc_y = r[1];

        // Note that in-bound kernels that touch the left and top margins are also considered as being out-of-bounds (see comparison: <= 0).
        // This is to fix the non-symmetric issue on the complex gridded matrix caused by the even matrix size.
        if ((kc_x - support) <= 0 || (kc_y - support) <= 0 || (kc_x + support) >= image_size || (kc_y + support) >= image_size) {
            good_vis[col] = 0;
        }
        col++;
    });
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
