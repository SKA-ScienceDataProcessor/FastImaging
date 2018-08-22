/** @file linear_interpolation.h
 *  @brief Linear interpolation functions
 */

#ifndef LINEAR_INTERPOLATION_H
#define LINEAR_INTERPOLATION_H

#include <armadillo>
#include <types.h>

real_t linear_interpolation(arma::Col<real_t>& x, arma::Col<real_t>& y, real_t& xi_p, int loidx);

#endif /* LINEAR_INTERPOLATION_H */
