#include "gaussian2d.h"
#include <math.h>

arma::Mat<real_t> Gaussian2D::operator()(arma::uword n_rows, arma::uword n_cols) const
{
    // Auxiliary calculations for 2D gaussian function
    const double cost = cos(theta);
    const double cost2 = cost * cost;
    const double sint = sin(theta);
    const double sint2 = sint * sint;
    const double sin2t = sin(2.0 * theta);
    const double xstd2 = x_stddev * x_stddev;
    const double ystd2 = y_stddev * y_stddev;
    const double a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
    const double b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
    const double c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));

    arma::Mat<real_t> model(n_rows, n_cols);

    for (arma::uword i = 0; i < n_cols; ++i) {
        for (arma::uword j = 0; j < n_rows; ++j) {
            const double xdiff = i - x_mean;
            const double ydiff = j - y_mean;
            model.at(j, i) = amplitude * exp(-(a * xdiff * xdiff
                                             + b * xdiff * ydiff
                                             + c * ydiff * ydiff));
        }
    }

    return model;
}
