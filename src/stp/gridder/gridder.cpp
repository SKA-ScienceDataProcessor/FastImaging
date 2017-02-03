#include "gridder.h"

namespace stp {

arma::uvec bounds_check_kernel_centre_locations(arma::imat& kernel_centre_indices, int support, int image_size)
{
    arma::uvec out_of_bounds_bool(kernel_centre_indices.n_rows);

    int col = 0;
    kernel_centre_indices.each_row([&out_of_bounds_bool, &image_size, &support, &col](arma::imat& r) {
        const int kc_x = r[0];
        const int kc_y = r[1];
        if (kc_x - support < 0 || kc_y - support < 0 || kc_x + support >= image_size || kc_y + support >= image_size) {
            out_of_bounds_bool[col] = 1;
        } else {
            out_of_bounds_bool[col] = 0;
        }
        col++;
    });

    return arma::find(out_of_bounds_bool == 0);
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

    return oversampled_coord;
}
}
