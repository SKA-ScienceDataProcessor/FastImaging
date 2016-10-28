#include "conv_func.h"

// Symmetric about the origin. Linearly declines from 1.0 at origin to 0.0 at **half_base_width**, zero thereafter.
arma::mat Triangle::operator() (const arma::mat& radius_in_pix, const double half_base_width, const double triangle_value) {
    // Calculate the max between triangle_value - (radius_in_pix / half_base_width) and a array of zeros with radius_in_pix size 
    return arma::max((triangle_value - abs(radius_in_pix) / half_base_width), zeros(size(radius_in_pix)));
}

// Symmetric about the origin. Valued 1.0 from origin to **half_base_width**, zero thereafter.
arma::mat TopHat::operator() (const arma::mat& radius_in_pix, const double half_base_width) {   
    // Construct of an array filled with zeros and siz of radius_in_pix.
    arma::mat result_array = arma::zeros(radius_in_pix.n_rows, radius_in_pix.n_cols);
    // Replace in result_array all elements that are less than half_base_width with ones.
    result_array.elem(find(abs(radius_in_pix) < half_base_width)).ones();
    return result_array;
}

// Sinc function (with truncation).
arma::mat Sinc::operator() (const arma::mat& radius_in_pix) {
    // Construct of an array filled with sine of PI vs radius_in_pix splited by PI vs radius_in_pix
    arma::mat result_array = (sin(arma::datum::pi * radius_in_pix) / (arma::datum::pi * radius_in_pix));
    // Replace in result_array all "nan" elements  with ones.
    result_array.replace(arma::datum::nan, 1);
    return result_array;
}

// Gaussian function (with truncation)  evaluates the function:: exp(-(x/w)**2)
// (Using the notation of Taylor 1998, p143, where x = u/delta_u and alpha==2. Default value of ``w=1``).
arma::mat Gaussian::operator() (const arma::mat& radius_in_pix, const double width) {
    // Calculate the exponential of radius_in_pix / with vs the diagonal matrix of radius_in_pix / width 
    return exp(-1 * ((radius_in_pix / width) * diagmat(radius_in_pix / width)));
}

// Gaussian times sinc function (with truncation).
arma::mat GaussianSinc::operator() (const arma::mat& radius_in_pix, const double width_normalization_gaussian, const double width_normalization_sinc) {
    Gaussian gaussian;
    Sinc sinc;
    // Multiply the result of the gaussian function by the diagonal matrix of sinc funtion result.
    return (gaussian(radius_in_pix, width_normalization_gaussian)) * diagmat(sinc((radius_in_pix / width_normalization_sinc)));
}
