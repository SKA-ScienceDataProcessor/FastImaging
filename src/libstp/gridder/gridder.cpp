#include "gridder.h"

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

arma::mat calculate_oversampled_kernel_indices(arma::mat subpixel_coord, const double oversampling)
{

    subpixel_coord = subpixel_coord * oversampling;
    subpixel_coord.for_each([](arma::mat::elem_type& val) {
        val = rint(val);
    });
    int range_max = oversampling / 2;
    int range_min = -1 * range_max;

    arma::uvec range_max_idx = find(subpixel_coord == (range_max + 1));
    arma::uvec range_min_idx = find(subpixel_coord == (range_min - 1));

    subpixel_coord.elem(find(subpixel_coord == (range_max + 1))).fill(range_max);
    subpixel_coord.elem(find(subpixel_coord == (range_min - 1))).fill(range_min);

    return subpixel_coord;
}
