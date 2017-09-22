#ifndef FIXTURES_H
#define FIXTURES_H

#include <armadillo>
#include <cfloat>
#include <stp.h>

/** @brief uncorrelated_gaussian_noise_background function
 *
 *  Returns a matrix of gaussian noise.
 *
 *  @param[in] rows (size_t): number of rows
 *  @param[in] cols (size_t): number of columns
 *  @param[in] sigma (double): noise standard deviation (default: 1.0)
 *  @param[in] mean (double): noise mean value (default: 0.0)
 *
 *  @return (arma::mat): gaussian noise background
 */
arma::Mat<real_t> uncorrelated_gaussian_noise_background(
    size_t rows,
    size_t cols,
    double sigma = 1.0,
    double mean = 0.0,
    long seed = -1)
{
    if (seed == -1) {
        arma::arma_rng::set_seed_random();
    } else {
        arma::arma_rng::set_seed(seed);
    }
    return (sigma * arma::randn<arma::Mat<real_t>>(rows, cols)) + mean;
}

/** @brief evaluate_model_on_pixel_grid function
 *
 *  Computes input model (e.g. gaussian2D) in a matrix defined by input x_rows and y_rows.
 *
 *  @param[in] y (double): number of rows
 *  @param[in] x (double): number of columns
 *  @param[in] model (double): input model to be computed
 *
 *  @return (arma::mat): computed model
 */
template <typename T>
arma::Mat<real_t> evaluate_model_on_pixel_grid(double y, double x, const T& model)
{
    return model(y, x);
}

#endif /* FIXTURES_H */
