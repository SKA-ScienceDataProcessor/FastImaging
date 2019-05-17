/**
 * @file gridder.cpp
 * @brief Implementation of the gridder functions.
 */

#include "gridder.h"
#include "../common/spharmonics.h"

namespace stp {

void convert_to_halfplane_visibilities(arma::mat& uv_lambda, arma::cx_mat& vis, int kernel_support, arma::Col<uint>& good_vis)
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

void convert_to_halfplane_visibilities(arma::mat& uv_lambda, arma::vec& w_lambda, arma::cx_mat& vis, int kernel_support, arma::Col<uint>& good_vis)
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

void average_w_planes(arma::mat w_lambda, const arma::Col<uint>& good_vis, uint num_wplanes, arma::Col<real_t>& w_avg_values, arma::uvec& w_planes_firstidx, bool median)
{
    arma::uword num_elems = good_vis.n_elem;
    arma::uword num_gvis = 0;
    // Count good visibilities
    for (size_t i = 0; i < good_vis.n_elem; ++i) {
        if (good_vis[i] != 0) {
            num_gvis++;
        }
    }

    if (median) {

        arma::uword begin = 0;
        for (arma::uword idx = 0; idx < num_wplanes; idx++) {

            // Because exact plane size is fractional, compute it at every iteration
            arma::uword plane_size = arma::uword(std::ceil(double(num_gvis - begin) / (num_wplanes - idx)));
            assert(plane_size >= 1);
            arma::uword idx_mid = 0;
            arma::uword middle = plane_size / 2;
            if ((begin + plane_size) > num_gvis) {
                middle = (num_gvis - begin) / 2;
            }

            // average w-plane components
            arma::uword k;
            arma::uword counter = 0;
            for (k = begin; (counter < plane_size) && (k < num_elems); k++) {
                if (good_vis(k) != 0) {
                    if (counter == middle) {
                        idx_mid = k;
                    }
                    counter++;
                }
            }
            if (counter > 0) {
                if ((plane_size % 2) == 0) {
                    // Even number
                    w_avg_values[idx] = real_t((w_lambda(idx_mid - 1) + w_lambda(idx_mid)) / 2.0);
                } else {
                    // Odd number
                    w_avg_values[idx] = real_t(w_lambda(idx_mid));
                }
            }
            else
                w_avg_values[idx] = 0.0;

            w_planes_firstidx[idx] = begin;
            begin = k;
        }

    } else {
        arma::uword begin = 0;
        for (arma::uword idx = 0; idx < num_wplanes; idx++) {

            // Because exact plane size is fractional, compute it at every iteration
            arma::uword plane_size = arma::uword(std::ceil(double(num_gvis - begin) / (num_wplanes - idx)));
            assert(plane_size >= 1);

            // average w-plane components
            double sum = 0;
            arma::uword k;
            arma::uword counter = 0;
            for (k = begin; (counter < plane_size) && (k < num_elems); k++) {
                if (good_vis(k) != 0) {
                    sum += w_lambda(k);
                    counter++;
                }
            }
            w_planes_firstidx[idx] = begin;
            begin = k;
            if (counter > 0)
                w_avg_values[idx] = real_t(sum / counter);
            else
                w_avg_values[idx] = 0.0;
        }
    }
}

void average_lha_planes(const arma::vec& lha, const arma::Col<uint>& good_vis, uint num_timesteps, arma::Col<real_t>& lha_planes, arma::ivec& vis_timesteps)
{
    // Find maximum and minimum lha values
    double min_time = lha[0];
    double max_time = lha[0];
    lha.for_each([&](const arma::vec::elem_type& val) {
        if (val < min_time) {
            min_time = val;
        }
        if (val > max_time) {
            max_time = val;
        }
    });

    // Define time intervals
    double tstep_size = (max_time - min_time) / num_timesteps;
    arma::vec time_intervals = arma::linspace(min_time - tstep_size / 2, max_time + tstep_size / 2, num_timesteps + 1);
    arma::uvec nvis_tstep_aux = arma::zeros<arma::uvec>(num_timesteps);
    lha_planes = arma::zeros<arma::Col<real_t>>(num_timesteps);

    // Assign each visibility to a time interval
    for (arma::uword i = 0; i < lha.n_elem; i++) {
        if (!good_vis[i])
            continue;

        double time_value = lha.at(i);
        uint tidx = 1;
        double interval_value = time_intervals.at(tidx);
        while (time_value > interval_value) {
            tidx++;
            if (tidx < (num_timesteps + 1)) {
                interval_value = time_intervals.at(tidx);
            } else {
                break;
            }
        }
        // Selected time step
        uint seltimestep_idx = tidx - 1;
        assert(seltimestep_idx < num_timesteps);
        vis_timesteps[i] = seltimestep_idx;
        // Auxiliary calcs for mean computation
        lha_planes.at(seltimestep_idx) += real_t(time_value);
        nvis_tstep_aux.at(seltimestep_idx)++;
    }
    // Compute mean values
    for (uint i = 0; i < num_timesteps; i++) {
        if (nvis_tstep_aux.at(i)) {
            lha_planes.at(i) /= real_t(nvis_tstep_aux.at(i));
        } else {
            lha_planes.at(i) = real_t(0.0);
        }
    }
}

void bounds_check_kernel_centre_locations(arma::Col<uint>& good_vis, const arma::Mat<int>& kernel_centre_on_grid, int image_size, int support)
{
    // Check bounds
    uint col = 0;
    kernel_centre_on_grid.each_row([&](const arma::Mat<int>& r) {
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

arma::Mat<int> calculate_oversampled_kernel_indices(arma::mat& subpixel_coord, uint oversampling)
{
    assert(arma::uvec(arma::find(arma::abs(subpixel_coord) > 0.5)).is_empty());

    arma::Mat<int> oversampled_coord(arma::size(subpixel_coord));

    int range_max = oversampling / 2;
    int range_min = -1 * range_max;

    for (arma::uword i = 0; i < subpixel_coord.n_elem; i++) {
        int val = int(rint(subpixel_coord.at(i) * oversampling));
        if (val > range_max) {
            val = range_max;
        } else {
            if (val < range_min) {
                val = range_min;
            }
        }
        oversampled_coord.at(i) = val;
    }

    return oversampled_coord;
}

arma::Mat<real_t> generate_a_kernel(const A_ProjectionPars& a_proj, const double fov, const int workarea_size, double rot_angle)
{
    arma::Mat<real_t> Akernel((size_t)workarea_size, (size_t)workarea_size);
    int degree = int((a_proj.pbeam_coefs.size() - 1) / 2);
    int workarea_centre = workarea_size / 2;

    tbb::parallel_for(tbb::blocked_range<int>(-workarea_centre, workarea_centre),
        [&](const tbb::blocked_range<int>& r) {
            int j_begin = r.begin(), j_end = r.end();
            for (int j = j_begin; j < j_end; ++j) {
                for (int i = -workarea_centre; i < workarea_centre; ++i) {
                    // Compute phi and theta
                    double j_rad = double(j) * fov / double(workarea_size);
                    double i_rad = double(i) * fov / double(workarea_size);
                    double theta = std::sqrt(j_rad * j_rad + i_rad * i_rad); // Polar angle
                    double phi = M_PI / 2.0;                                 // Azimuth angle
                    if (j != 0) {
                        phi = std::atan(i_rad / j_rad);
                    }
                    // Init Akernel
                    Akernel(size_t(j + workarea_centre), size_t(i + workarea_centre)) = 0;
                    // Compute each spherical harmonic, get abs and sum
                    for (int m = -degree; m <= degree; m++) { // SH order range
                        Akernel(size_t(j + workarea_centre), size_t(i + workarea_centre)) += real_t(std::abs(sh::EvalSH(degree, m, phi + rot_angle, theta) * a_proj.pbeam_coefs[degree + m]));
                    }
                    // Invert Akernel value
                    Akernel(size_t(j + workarea_centre), size_t(i + workarea_centre)) = 1.0 / Akernel(size_t(j + workarea_centre), size_t(i + workarea_centre));
                }
            }
        });

    return Akernel;
}
}
