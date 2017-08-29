/**
 * @file imager.cpp
 * @brief Implementation of the imager functions.
 */

#include "imager.h"
#include <armadillo>
#include <cassert>

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
}
