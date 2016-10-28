#include "gridder.h"

arma::uvec bounds_check_kernel_centre_locations(arma::mat uv, arma::mat kernel_centre_indices, int support, int image_size, bool raise_if_bad) {
    arma::vec out_of_bounds_bool = arma::zeros<arma::vec>(kernel_centre_indices.n_rows);

    int col(0);
    kernel_centre_indices.each_row([&out_of_bounds_bool, &image_size, &support, &col](arma::mat &r) {
        if(r[0] - support < 0 ||
        r[1] - support < 0 ||
        r[0] + support >= image_size ||
        r[1] + support >= image_size )
        {
            out_of_bounds_bool[col] = 1;
        }
        col++;
    });

    return arma::find(out_of_bounds_bool == 0);
}
