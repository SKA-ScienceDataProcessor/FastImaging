#ifndef GAUSSIAN2D_H
#define GAUSSIAN2D_H

#include <armadillo>
#include <stp.h>

class Gaussian2D {

public:
    double amplitude;
    double x_mean;
    double y_mean;
    double x_stddev;
    double y_stddev;
    double theta;

    Gaussian2D(
        double in_x_mean = 0.0,
        double in_y_mean = 0.0,
        double in_amplitude = 1.0,
        double in_x_stddev = 1.0,
        double in_y_stddev = 1.0,
        double in_theta = 0.0)
        : x_mean(in_x_mean)
        , y_mean(in_y_mean)
        , amplitude(in_amplitude)
        , x_stddev(in_x_stddev)
        , y_stddev(in_y_stddev)
        , theta(in_theta)
    {
    }
    /**
     * @brief Computes gaussian 2D model
     *
     * Receives n_rows and n_cols of the Gaussian 2D model matrix to be generated.
     *
     * @param[in] n_rows (arma::uword): number of rows
     * @param[in] n_cols (arma::uword): number of columns
     *
     * @return (arma::mat) The calculated model
     */
    arma::Mat<real_t> operator()(arma::uword n_rows, arma::uword n_cols) const;
};

typedef Gaussian2D gaussian_point_source;

#endif /* GAUSSIAN2D_H */
