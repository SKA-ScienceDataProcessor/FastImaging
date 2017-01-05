#include "gaussian2d.h"

namespace stp {

arma::mat Gaussian2D::operator()(arma::mat& xgrid, arma::mat& ygrid) const
{
    arma::uword n_rows = xgrid.n_rows;
    arma::uword n_cols = xgrid.n_cols;
    double a((cos(theta) * cos(theta) / (2 * x_stddev * x_stddev)) + (sin(theta) * sin(theta) / (2 * y_stddev * y_stddev)));
    double b((sin(2 * theta) / (2 * x_stddev * x_stddev)) - (sin(2 * theta) / (2 * y_stddev * y_stddev)));
    double c((sin(theta) * sin(theta) / (2 * x_stddev * x_stddev)) + (cos(theta) * cos(theta) / (2 * y_stddev * y_stddev)));

    arma::mat model(n_rows, n_cols);
    arma::mat::iterator itx = xgrid.begin();
    arma::mat::iterator ity = ygrid.begin();

    model.for_each([&itx, &ity, this, &a, &b, &c](arma::mat::elem_type& val) {
        double x = (*itx);
        double y = (*ity);
        val = amplitude * exp(-a * (x - x_mean) * (x - x_mean)
                              - b * (x - x_mean) * (y - y_mean)
                              - c * (y - y_mean) * (y - y_mean));
        ++itx;
        ++ity;
    });

    return model;
}
}
