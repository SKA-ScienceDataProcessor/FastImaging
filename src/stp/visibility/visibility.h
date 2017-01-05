/**
* @file visibility.h
* Contains the prototypes and implementation of visibility functions
*/

#ifndef VISIBILITY_H
#define VISIBILITY_H

#include <armadillo>
#include <cmath>
#include <complex>
#include <random>

namespace stp {

/**
 * @brief The Skycoord struct
 */
struct SkyCoord {
    SkyCoord() = default;
    /**
     * @brief SkyCoord constructor
     *
     * @param ra (double): The angular offset in the longitude direction (i.e., RA for equatorial coordinates) [degrees]
     * @param dec (double): The angular offset in the latitude direction (i.e., Dec for equatorial coordinates) [degrees]
     */
    SkyCoord(double ra, double dec)
        : _ra(ra)
        , _dec(dec)
    {
    }
    double _ra;
    double _dec;
};

/**
 * @brief The SkyRegion struct
 *
 *  Defines a circular region of the sky.
 *
 */
struct SkyRegion {
    SkyRegion() = default;
    /**
     * @brief SkyRegion constructor
     *
     * @param centre (SkyCoord): Sky region centre.
     * @param radius (double): Sky region radius.
     */
    SkyRegion(SkyCoord& centre, double radius)
        : _centre(centre)
        , _radius(radius)
    {
    }
    SkyCoord _centre;
    double _radius;
};

/**
 * @brief The SkySource struct
 *
 *  Basic point source w/ flux modelled at a single frequency
 *
 */
struct SkySource {
    SkySource() = default;
    /**
     * @brief SkySource constructor
     *
     * @param position (SkyCoord): Sky-coordinates of source.
     * @param flux (double): Source flux at measured frequency.
     */
    SkySource(SkyCoord& position, double flux)
        : _position(position)
        , _flux(flux)
    {
    }
    SkyCoord _position;
    double _flux;
};

/** @brief l_cosine function
 *
 * Convert a coordinate in RA,Dec into a direction cosine l (RA-direction)
 *
 * @param[in] ra (double): Source location [rad]
 * @param[in] dec (double): Source location [dec]
 * @param[in] ra0 (double): RA centre of the field [rad]
 *
 * @return l (double): Direction cosine
 */
double l_cosine(double ra, double dec, double ra0);

/** @brief m_cosine function
 *
 * Convert a coordinate in RA,Dec into a direction cosine m (Dec-direction)
 *
 * @param[in] ra (double): Source location [rad]
 * @param[in] dec (double): Source location [dec]
 * @param[in] ra0 (double): Centre of the field [rad]
 * @param[in] dec0 (double): Centre of the field [dec]
 *
 * @return m (double): Direction cosine
 */
double m_cosine(double ra, double dec, double ra0, double dec0);

/** @brief visibilities_for_point_source function
 *
 * Simulate visibilities for point source. Calculate visibilities for a source located at
 * angular position (l,m) relative to observed phase centre as used for calculating
 * baselines in UVW space. Note that point source is delta function, therefore FT relationship
 * becomes an exponential, evaluated at (uvw.lmn)
 *
 * @param[in] dist_uvw (arma::mat): Array of 3-vectors representing baselines in UVW space [lambda].
 * @param[in] l (double): Direction cosines in RA direction
 * @param[in] m (double): Direction cosines in Dec direction
 * @param[in] flux (double): Flux [Jy]
 *
 * @return vis (arma::cx_mat): Array of complex visibilities
 */
arma::cx_mat visibilities_for_point_source(arma::mat& dist_uvw, double l, double m, double flux);

/** @brief visibilities_for_source_list function
 *
 * Generate noise-free visibilities from UVW baselines and point-sources.
 *
 * @param[in] pointing_centre (SkyCoord): Pointing centre
 * @param[in] source_list (std::vector<SkySource>): List of list of :class:`fastimgproto.skymodel.helpers.SkySource`
 * @param[in] uvw (arma::cx_mat): UVW baselines (units of lambda). Numpy array shape: (n_baselines, 3)
 *
 * @return vis (arma::cx_mat): Complex visbilities sum for each baseline. Numpy array shape: (n_baselines)
 */
arma::cx_mat visibilities_for_source_list(SkyCoord& pointing_centre, std::vector<SkySource> source_list, arma::mat& uvw);

/** @brief add_gaussian_noise function
 *
 * Add random Gaussian distributed noise to the visibilities.
 * Adds jointly-normal (i.e. independent) Gaussian noise to both the real and imaginary components of the visibilities.
 *
 * @param[in] noise_level (double): This defines the std. dev. / sigma of the Gaussian distribution.
 * @param[in] vis (arma::cx_mat): The array of (noise-free) complex visibilities.
 * @param[in] seed (double):  Optional - can be used to seed the random number generator to ensure reproducible results.
 *
 * @return vis (arma::cx_mat): Visibilities with complex Gaussian noise added.
 */
arma::cx_mat add_gaussian_noise(double noise_level, arma::cx_mat vis, double seed = 0);
}

#endif /* VISIBILITY_H */
