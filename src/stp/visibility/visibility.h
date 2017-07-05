/**
* @file visibility.h
* Contains the prototypes of visibility functions
*/

#ifndef VISIBILITY_H
#define VISIBILITY_H

#include <armadillo>
#include <cmath>
#include <complex>
#include <random>

namespace stp {

/**
 * @brief Convert value in degrees to radians
 *
 * @param[in] value (double): value in degrees
 *
 * @return (double): value in radians
 */
constexpr double degree_to_rad(double value);

/**
 * @brief Simulate visibilities for point source.
 *
 * Calculate visibilities for a source located at angular position (l,m) relative to observed
 * phase centre as used for calculating baselines in UVW space.
 * Note that point source is delta function, therefore FT relationship
 * becomes an exponential, evaluated at (uvw.lmn)
 *
 * @param[in] dist_uvw (arma::mat): Array of 3-vectors representing baselines in UVW space [lambda].
 * @param[in] l (double): Direction cosines in RA direction
 * @param[in] m (double): Direction cosines in Dec direction
 * @param[in] flux (double): Flux [Jy]
 *
 * @return arma::cx_mat: Array of complex visibilities
 */
arma::cx_mat visibilities_for_point_source(arma::mat& dist_uvw, double l, double m, double flux);

/**
 * @brief Generate a set of model visibilities given a skymodel and UVW-baselines.
 *
 * @param[in] skymodel (arma::mat): The local skymodel. Array of triples [l,m,flux_jy], where 'l,m' are the
 *                                  directional cosines for this source, and 'flux_jy' is flux in Janskys.
 *                                  Shape: (n_baselines, 3)
 * @param[in] uvw_baselines (arma::mat): UVW baselines (units of lambda).
 *                                       Shape: (n_baselines, 3)
 *
 * @return arma::cx_mat: Complex visibilities sum for each baseline. Length: n_baselines.
 */
arma::cx_mat generate_visibilities_from_local_skymodel(arma::mat& skymodel, arma::mat& uvw_baselines);
}

#endif /* VISIBILITY_H */
