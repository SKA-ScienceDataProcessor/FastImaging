#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

#include <armadillo>

namespace stp {

double vector_accumulate(arma::vec& v);
double vector_accumulate_parallel(arma::vec& v);

double vector_mean(arma::vec& v);
double vector_mean_parallel(arma::vec& v);
double vector_mean_robust(arma::vec& v);
double vector_mean_robust_parallel(arma::vec& v);

double vector_stddev(arma::vec& v, double mean = arma::datum::nan);
double vector_stddev_parallel(arma::vec& v, double mean = arma::datum::nan);
double vector_stddev_robust(arma::vec& v, double mean = arma::datum::nan);
double vector_stddev_robust_parallel(arma::vec& v, double mean = arma::datum::nan);

arma::cx_mat matrix_shift(const arma::cx_mat& in, const int length, const int dim);
}

#endif /* VECTOR_MATH_H */
