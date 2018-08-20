/**
 * @file conv_func.cpp
 * @brief Implementation of classes and functions of convolution methods.
 */

#include "conv_func.h"

namespace stp {

// Symmetric about the origin. Linearly declines from 1.0 at origin to 0.0 at **half_base_width**, zero thereafter.
arma::vec Triangle::operator()(const arma::vec& radius_in_pix) const
{
    // Calculate the max between triangle_value - (radius_in_pix / half_base_width) and a array of zeros with radius_in_pix size
    return arma::max((1.0 - arma::abs(radius_in_pix) / _half_base_width), arma::zeros(arma::size(radius_in_pix)));
}

// Triangle grid correction factor
arma::vec Triangle::gcf(const arma::vec& radius) const
{
    assert(false);
    return radius;
}

// Symmetric about the origin. Valued 1.0 from origin to **half_base_width**, zero thereafter.
arma::vec TopHat::operator()(const arma::vec& radius_in_pix) const
{
    // Construct of an array filled with zeros and size of radius_in_pix.
    arma::vec result(arma::zeros(radius_in_pix.n_rows, radius_in_pix.n_cols));

    // Replace in result all elements that are less than half_base_width with ones.
    result.elem(arma::find(arma::abs(radius_in_pix) < _half_base_width)).ones();

    return result;
}

// TopHat grid correction factor
arma::vec TopHat::gcf(const arma::vec& radius) const
{
    assert(false);
    return radius;
}

arma::vec Sinc::operator()(const arma::vec& radius_in_pix) const
{
    // Construction of an array filled with sine of PI vs radius_in_pix split by PI vs radius_in_pix
    arma::vec normalized_radius_in_pix(radius_in_pix * (arma::datum::pi / _width_normalization));
    // Replace positions where normalized_radius_in_pix is zero by 1.0e-20 (to avoid nan values).
    normalized_radius_in_pix.replace(0.0, 1.0e-20);

    arma::vec result(arma::sin(normalized_radius_in_pix) / normalized_radius_in_pix);

    // Truncate the result if required
    if (_trunc > 0.0) {
        result.elem(find(arma::abs(radius_in_pix) > _trunc)).zeros();
    }

    return result;
}

// Sinc grid correction factor
arma::vec Sinc::gcf(const arma::vec& radius) const
{
    arma::vec result = arma::zeros(radius.n_elem);
    result.elem(find(arma::abs(radius) < (1.0 * _width_normalization))).ones();
    return result;
}

// Gaussian function (with truncation)  evaluates the function:: exp(-(x/w)**2)
// (Using the notation of Taylor 1998, p143, where x = u/delta_u and alpha==2. Default value of ``w=1``).
arma::vec Gaussian::operator()(const arma::vec& radius_in_pix) const
{
    // f(x) = exp(-(x/w)**2)
    arma::vec result(arma::exp((-1.0 / (_width_normalization * _width_normalization)) * arma::square(radius_in_pix)));

    // Truncate the result if required
    if (_trunc > 0.0) {
        result.elem(arma::find(arma::abs(radius_in_pix) > _trunc)).zeros();
    }

    return result;
}

// Gaussian grid correction factor
arma::vec Gaussian::gcf(const arma::vec& radius) const
{
    arma::vec result(arma::exp(-1.0 * arma::square(radius * M_PI / (_width_normalization * 2))));
    return result;
}

// Gaussian times sinc function
arma::vec GaussianSinc::operator()(const arma::vec& radius_in_pix) const
{
    // f(x) = exp(-(x/w1)**2) * sinc(x/w2)
    arma::vec result(_gaussian(radius_in_pix) % _sinc(radius_in_pix));

    // Truncate the result if required
    if (_trunc > 0.0) {
        result.elem(arma::find(arma::abs(radius_in_pix) > _trunc)).zeros();
    }

    return result;
}

// GaussianSinc grid correction factor
arma::vec GaussianSinc::gcf(const arma::vec& radius) const
{
    assert(false);
    return radius;
}

arma::vec __grdsf(arma::vec nu)
{
    /* Calculate PSWF using an old SDE routine:
* Find Spheroidal function with M = 6, alpha = 1 using the rational approximations discussed by Fred Schwab in 'Indirect Imaging'.
* This routine was checked against Fred's SPHFN routine, and agreed to about the 7th significant digit.
* The gridding function is (1-NU**2)*GRDSF(NU) where NU is the distance to the edge. The grid correction function is just 1/GRDSF(NU) where NU
* is now the distance to the edge of the image.
*/
    arma::mat p = { { 8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1 },
        { 4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2 } };
    arma::mat q = { { 1.0000000e0, 8.212018e-1, 2.078043e-1 },
        { 1.0000000e0, 9.599102e-1, 2.918724e-1 } };

    // auxiliary
    int nu_size = nu.n_elem;
    int n_p = p.n_cols;
    int n_q = q.n_cols;
    arma::vec nuend(nu_size);
    arma::vec top(nu_size);
    arma::vec bot(nu_size);
    arma::uvec part(nu_size);

    // return vec
    arma::vec grdsf(nu_size);

    // module
    nu = arma::abs(nu);

    for (int i = 0; i < nu_size; i++) {
        if ((nu[i] >= 0.0) && (nu[i] < 0.75)) {
            part[i] = 0;
            top[i] = p.at(0, 0);
            bot[i] = q.at(0, 0);
            nuend[i] = 0.75;
        } else if ((nu[i] >= 0.75) && (nu[i] <= 1.0)) {
            part[i] = 1;
            top[i] = p.at(1, 0);
            bot[i] = q.at(1, 0);
            nuend[i] = 1.0;
        } else {
            part[i] = 0;
            top[i] = p.at(0, 0);
            bot[i] = q.at(0, 0);
            nuend[i] = 0.0;
        }
    }

    arma::vec delnusq(arma::pow(nu, 2) - arma::pow(nuend, 2));

    for (int k = 1; k < n_p; k++)
        for (int x = 0; x < nu_size; x++) {
            top[x] += p.at(part[x], k) * std::pow(delnusq[x], k);
        }

    for (int k = 1; k < n_q; k++)
        for (int x = 0; x < nu_size; x++) {
            bot[x] += q.at(part[x], k) * std::pow(delnusq[x], k);
        }

    for (int i = 0; i < nu_size; i++) {
        if (bot[i] > 0.0) {
            grdsf[i] = top[i] / bot[i];
        } else if (std::abs(nu[i]) > 1.0) {
            grdsf[i] = 0.0;
        } else {
            grdsf[i] = 0.0;
        }
    }

    return grdsf;
}

// PSWF function
arma::vec PSWF::operator()(const arma::vec& radius_in_pix) const
{
    arma::vec nu = radius_in_pix / _trunc;
    arma::vec result(__grdsf(nu) % (1 - (arma::pow(nu, 2))));

    // Truncate the result if required
    if (_trunc > 0.0) {
        result.elem(arma::find(arma::abs(radius_in_pix) > _trunc)).zeros();
    }

    return result;
}

// PSWF grid correction factor
arma::vec PSWF::gcf(const arma::vec& radius) const
{
    return __grdsf(radius);
}
}
