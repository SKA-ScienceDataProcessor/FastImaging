/** @file linear_interpolation.cpp
 *  @brief Implementation of linear interpolation functions
 */

#include "linear_interpolation.h"

real_t linear_interpolation(arma::Col<real_t>& x, arma::Col<real_t>& y, real_t& xi_p, int loidx)
{
    int upidx = loidx + 1;
    real_t weighted_x = (xi_p - x(loidx)) / (x(upidx) - x(loidx));
    return y(loidx) * (1 - weighted_x) + y(upidx) * weighted_x;
}
