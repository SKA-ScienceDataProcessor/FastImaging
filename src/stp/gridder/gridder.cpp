#include "gridder.h"

namespace stp {

arma::uvec bounds_check_kernel_centre_locations(arma::mat kernel_centre_indices, int support, int image_size)
{
    arma::vec out_of_bounds_bool = arma::zeros<arma::vec>(kernel_centre_indices.n_rows);

    int col(0);
    kernel_centre_indices.each_row([&out_of_bounds_bool, &image_size, &support, &col](arma::mat& r) {
        if (r[0] - support < 0 || r[1] - support < 0 || r[0] + support >= image_size || r[1] + support >= image_size) {
            out_of_bounds_bool[col] = 1;
        }
        col++;
    });

    return arma::find(out_of_bounds_bool == 0);
}

arma::mat calculate_oversampled_kernel_indices(arma::mat subpixel_coord, const int oversampling_val)
{

    assert(arma::uvec(arma::find(arma::abs(subpixel_coord) > 0.5)).is_empty());

    subpixel_coord = subpixel_coord * oversampling_val;
    subpixel_coord.transform([](arma::mat::elem_type& val) {
        return rint(val);
    });

    int range_max = oversampling_val / 2;
    int range_min = -1 * range_max;

    subpixel_coord.elem(find(subpixel_coord == (range_max + 1))).fill(range_max);
    subpixel_coord.elem(find(subpixel_coord == (range_min - 1))).fill(range_min);

    return subpixel_coord;
}
}
