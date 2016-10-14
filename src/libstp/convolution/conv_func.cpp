#include "conv_func.h"

mat make_conv_func_triangle(const double half_base_width, const mat& radius_in_pix, const double triangle_value) {
    return arma::max((triangle_value - abs(radius_in_pix) / half_base_width), zeros(size(radius_in_pix)));
} // make_conv_func_triangle

mat make_conv_func_tophat(const double half_base_width, const mat& radius_in_pix) {
    mat result_array = zeros(radius_in_pix.n_rows, radius_in_pix.n_cols);
    result_array.elem(find(abs(radius_in_pix) < half_base_width)).ones();
    return result_array;
} // make_conv_func_tophat

mat make_conv_func_sinc(const mat& radius_in_pix) {
    mat result_array = (sin(datum::pi * radius_in_pix) / (datum::pi * radius_in_pix));
    result_array.replace(datum::nan, 1);
    return result_array;
} // make_conv_func_sinc

mat make_conv_func_gaussian(const mat& radius_in_pix, const double w1) {
    return exp(-1 * ((radius_in_pix / w1) * diagmat(radius_in_pix / w1)));
} // make_conv_func_gaussian

mat make_conv_func_gaussian_sinc(const mat& radius_in_pix, const double w1, const double w2) {
    return make_conv_func_gaussian(radius_in_pix, w1) * diagmat(make_conv_func_sinc((radius_in_pix / w2)));
} // make_conv_func_gaussian_sinc
