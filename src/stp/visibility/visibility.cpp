/**
* @file visibility.cpp
* Contains the prototypes and implementation of visibility functions
*/

#include "visibility.h"

namespace stp {

constexpr double
degree_to_rad(double value)
{
    return (value * M_PI / 180.0);
}

double l_cosine(double ra, double dec, double ra0)
{
    return cos(dec) * sin(ra - ra0);
}

double m_cosine(double ra, double dec, double ra0, double dec0)
{
    return ((sin(dec) * cos(dec0)) - (cos(dec) * sin(dec0) * cos(ra - ra0)));
}

arma::cx_mat visibilities_for_point_source(arma::mat& dist_uvw, double l, double m, double flux)
{

    // Component of source vector along n-axis / w-axis
    // (Doubles as flux attenuation factor due to projection effects)
    double src_n = sqrt(1 - l * l - m * m);

    arma::vec src_offset = { l, m, src_n - 1 };
    src_offset = -1 * src_offset;

    arma::cx_mat vis = flux * src_n * arma::exp(arma::cx_double(0.0, -2.0) * arma::datum::pi * (dist_uvw * src_offset));

    return vis;
}

arma::cx_mat visibilities_for_source_list(SkyCoord& pointing_centre, std::vector<SkySource> source_list, arma::mat& uvw)
{
    arma::cx_mat sumvis;
    sumvis.zeros(uvw.n_rows, 1);

    for (uint i = 0; i < source_list.size(); i++) {
        SkyCoord& sp = source_list.at(i)._position;

        double l = l_cosine(degree_to_rad(sp._ra), degree_to_rad(sp._dec), degree_to_rad(pointing_centre._ra));
        double m = m_cosine(degree_to_rad(sp._ra), degree_to_rad(sp._dec), degree_to_rad(pointing_centre._ra),
            degree_to_rad(pointing_centre._dec));

        arma::cx_mat vis = visibilities_for_point_source(uvw, l, m, source_list.at(i)._flux);
        sumvis += vis;
    }

    return sumvis;
}

arma::cx_mat add_gaussian_noise(double noise_level, arma::cx_mat vis, double seed)
{
    arma::arma_rng::set_seed(seed);

    arma::cx_mat noise;
    noise.randn(size(vis));
    noise = noise * noise_level;
    vis += noise;

    return vis;
}
}
