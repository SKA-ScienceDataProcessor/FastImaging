#include "gaussian2d.h"

/**
 * @brief Computes gaussian 2D model
 *
 * Receives xgrid and ygrid matrices with (x,y) coordinates to be used.
 * Gaussian 2D model is stored in model matrix.
 *
 * @param[in] xgrid (arma::mat) : X grid coordinates
 * @param[in] ygrid (arma::mat) : Y grid coordinates
 *
 */
arma::mat Gaussian2D::operator()(arma::mat& xgrid, arma::mat& ygrid) const
{
    arma::uword n_rows = xgrid.n_rows;
    arma::uword n_cols = xgrid.n_cols;
    double a((cos(theta) * cos(theta) / (2 * x_stddev * x_stddev)) + (sin(theta) * sin(theta) / (2 * y_stddev * y_stddev)));
    double b((sin(2 * theta) / (2 * x_stddev * x_stddev)) - (sin(2 * theta) / (2 * y_stddev * y_stddev)));
    double c((sin(theta) * sin(theta) / (2 * x_stddev * x_stddev)) + (cos(theta) * cos(theta) / (2 * y_stddev * y_stddev)));

    arma::mat model(n_rows, n_cols);
    for (arma::uword j = 0; j < n_cols; j++) {
        for (arma::uword i = 0; i < n_rows; i++) {
            model.at(i, j) = amplitude * exp(-a * (xgrid(i, j) - x_mean) * (xgrid(i, j) - x_mean)
                                             - b * (xgrid(i, j) - x_mean) * (ygrid(i, j) - y_mean)
                                             - c * (ygrid(i, j) - y_mean) * (ygrid(i, j) - y_mean));
        }
    }

    return model;
}
