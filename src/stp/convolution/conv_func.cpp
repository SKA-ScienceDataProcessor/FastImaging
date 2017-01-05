#include "conv_func.h"

namespace stp {

// Symmetric about the origin. Linearly declines from 1.0 at origin to 0.0 at **half_base_width**, zero thereafter.
arma::mat Triangle::operator()(const arma::mat& radius_in_pix) const
{
    // Calculate the max between triangle_value - (radius_in_pix / half_base_width) and a array of zeros with radius_in_pix size
    return arma::max((_triangle_value - arma::abs(radius_in_pix) / _half_base_width), arma::zeros(arma::size(radius_in_pix)));
}

// Symmetric about the origin. Valued 1.0 from origin to **half_base_width**, zero thereafter.
arma::mat TopHat::operator()(const arma::mat& radius_in_pix) const
{
    // Construct of an array filled with zeros and size of radius_in_pix.
    arma::mat result(arma::zeros(radius_in_pix.n_rows, radius_in_pix.n_cols));

    // Replace in result all elements that are less than half_base_width with ones.
    result.elem(arma::find(arma::abs(radius_in_pix) < _half_base_width)).ones();

    return result;
}

arma::mat Sinc::operator()(const arma::mat& radius_in_pix) const
{
    // Construction of an array filled with sine of PI vs radius_in_pix split by PI vs radius_in_pix
    arma::mat normalized_radius_in_pix(radius_in_pix * (arma::datum::pi / _width_normalization));
    arma::mat result(arma::sin(normalized_radius_in_pix) / normalized_radius_in_pix);

    // Replace in result_array all "nan" elements  with ones.
    result.replace(arma::datum::nan, 1.0);

    // Truncate the result if required
    if (_truncate == true) {
        result.elem(find(arma::abs(radius_in_pix) >= _threshold)).zeros();
    }

    return result;
}

// Gaussian function (with truncation)  evaluates the function:: exp(-(x/w)**2)
// (Using the notation of Taylor 1998, p143, where x = u/delta_u and alpha==2. Default value of ``w=1``).
arma::mat Gaussian::operator()(const arma::mat& radius_in_pix) const
{
    // f(x) = exp(-(x/w)**2)
    arma::mat result(arma::exp((-1.0 / (_width_normalization * _width_normalization)) * arma::square(radius_in_pix)));

    // Truncate the result if required
    if (_truncate == true) {
        result.elem(arma::find(arma::abs(radius_in_pix) >= _threshold)).zeros();
    }

    return result;
}

// Gaussian times sinc function
arma::mat GaussianSinc::operator()(const arma::mat& radius_in_pix) const
{
    // f(x) = exp(-(x/w1)**2) * sinc(x/w2)
    arma::mat result(_gaussian(radius_in_pix) % _sinc(radius_in_pix));

    // Truncate the result if required
    if (_truncate == true) {
        result.elem(arma::find(arma::abs(radius_in_pix) >= _threshold)).zeros();
    }

    return result;
}
}
