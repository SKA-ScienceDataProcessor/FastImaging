#include "kernel_func.h"

mat make_top_hat_kernel_array(const int support, const mat& offset = { 0.0, 0.0 }, const double oversampling = 1.0, const double half_base_width = 0) {
    int array_size = 2 * support * oversampling + 1;

    mat distance_vec = (linspace(0, array_size - 1, array_size) / oversampling);

    vec x_kernel_coeffs = make_conv_func_tophat(half_base_width, (distance_vec - (support + offset[0])));
    vec y_kernel_coeffs = make_conv_func_tophat(half_base_width, (distance_vec - (support + offset[1])));

    return repmat(y_kernel_coeffs, 1, array_size) * diagmat(x_kernel_coeffs);
} // make_top_hat_kernel_array

mat make_triangle_kernel_array(const int support, const mat& offset = { 0.0, 0.0 }, const double oversampling = 1, const double half_base_width = 0) {
    int array_size = 2 * support * oversampling + 1;

    mat distance_vec = (linspace(0, array_size - 1, array_size) / oversampling);

    vec x_kernel_coeffs = make_conv_func_triangle(half_base_width, (distance_vec - (support + offset[0])), 1.0);
    vec y_kernel_coeffs = make_conv_func_triangle(half_base_width, (distance_vec - (support + offset[1])), 1.0);

    return repmat(y_kernel_coeffs, 1, array_size) * diagmat(x_kernel_coeffs);
} // make_triangle_kernel_array
