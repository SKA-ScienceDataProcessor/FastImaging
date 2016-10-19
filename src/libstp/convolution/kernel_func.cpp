#include "kernel_func.h"

mat make_top_hat_kernel_array(const int support, const mat& offset = { 0.0, 0.0 }, double oversampling = false, const double half_base_width = 0, double pad = false, bool normalize = false) {

   if (!oversampling) {
        oversampling = 1.0;
    }

    if (pad) {
        pad = 1.0;
    } else {
        pad = 0.0;
    }

    int array_size = 2 * (support + pad) * oversampling + 1;
    int centre_idx = (support + pad) * oversampling;

    mat distance_vec = ((linspace(0, array_size - 1, array_size) - centre_idx) / oversampling);

    vec x_kernel_coeffs = make_conv_func_tophat(half_base_width, (distance_vec - offset[0]));
    vec y_kernel_coeffs = make_conv_func_tophat(half_base_width, (distance_vec - offset[1]));
    

    mat result = repmat(y_kernel_coeffs, 1, array_size) * diagmat(x_kernel_coeffs);

    if (normalize){
        return result /= accu(result);
    }

    return result;
} // make_top_hat_kernel_array

mat make_triangle_kernel_array(const int support, const mat& offset = { 0.0, 0.0 }, double oversampling = 1, const double half_base_width = 0, double pad = false, bool normalize = false) { 
    
    if (!oversampling) {
        oversampling = 1.0;
    }

    if (pad) {
        pad = 1.0;
    } else {
        pad = 0.0;
    }

    int array_size = 2 * support * oversampling + 1;

    mat distance_vec = (linspace(0, array_size - 1, array_size) / oversampling);

    vec x_kernel_coeffs = make_conv_func_triangle(half_base_width, (distance_vec - (support + offset[0])), 1.0);
    vec y_kernel_coeffs = make_conv_func_triangle(half_base_width, (distance_vec - (support + offset[1])), 1.0);

    mat result = repmat(y_kernel_coeffs, 1, array_size) * diagmat(x_kernel_coeffs);

    if (normalize){
        return result /= accu(result);
    }

    return result;
} // make_triangle_kernel_array
