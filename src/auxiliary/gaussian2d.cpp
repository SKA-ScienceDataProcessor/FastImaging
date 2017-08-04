#include "gaussian2d.h"
#include <math.h>

arma::Mat<real_t> Gaussian2D::operator()(arma::uword n_rows, arma::uword n_cols) const
{
    double a = (cos(theta) * cos(theta) / (2 * x_stddev * x_stddev)) + (sin(theta) * sin(theta) / (2 * y_stddev * y_stddev));
    double b = (sin(2 * theta) / (2 * x_stddev * x_stddev)) - (sin(2 * theta) / (2 * y_stddev * y_stddev));
    double c = (sin(theta) * sin(theta) / (2 * x_stddev * x_stddev)) + (cos(theta) * cos(theta) / (2 * y_stddev * y_stddev));

    arma::Mat<real_t> model(n_rows, n_cols);

    for (arma::uword i = 0; i < n_cols; ++i) {
        for (arma::uword j = 0; j < n_rows; ++j) {
            model.at(j, i) = amplitude * std::exp(-a * (i - x_mean) * (i - x_mean)
                                             - 2 * b * (i - x_mean) * (j - y_mean)
                                             - c * (j - y_mean) * (j - y_mean));
        }
    }

    return model;
}
