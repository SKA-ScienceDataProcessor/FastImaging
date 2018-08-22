#include <gtest/gtest.h>
#include <linear_interpolation.h>
#include <stp.h>

using namespace stp;

TEST(LinearInterpolation, Correctness)
{
    size_t length = 1024;
    arma::Col<real_t> x, y, xi, yi, yi_arma, yi_spline;

    size_t length_interp = length * 8;
    x = arma::regspace<arma::Col<real_t>>(0, length - 1);
    y = arma::randu<arma::Col<real_t>>(arma::size(x));
    xi = arma::sort(arma::randu<arma::Col<real_t>>(length_interp) * (length - 2));

    yi.set_size(arma::size(xi));
    yi_arma.set_size(arma::size(xi));
    yi_spline.set_size(arma::size(xi));

    // Linear interpolation function
    for (size_t i = 0; i < xi.n_elem; ++i) {
        int lower_idx = int(xi[i]);
        yi[i] = linear_interpolation(x, y, xi[i], lower_idx);
    }

    // Armadillo
    arma::interp1(x, y, xi, yi_arma, "*linear");

    // Linear splines
    tk::spline<false> m_spline;
    m_spline.set_points(x, y, false);
    for (size_t i = 0; i < xi.n_elem; ++i) {
        int lower_idx = int(xi[i]);
        yi_spline[i] = m_spline(xi[i], lower_idx, x[lower_idx], y[lower_idx]);
    }

    EXPECT_TRUE(!arma::any(arma::abs(yi - yi_arma) > fptolerance));
    EXPECT_TRUE(!arma::any(arma::abs(yi - yi_spline) > fptolerance));
}
