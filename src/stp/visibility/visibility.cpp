/**
* @file visibility.cpp
* Contains the prototypes and implementation of visibility functions
*/

#include "visibility.h"

namespace stp {

constexpr double degree_to_rad(double value)
{
    return (value * M_PI / 180.0);
}

arma::cx_mat visibilities_for_point_source(arma::mat& dist_uvw, double l, double m, double flux)
{

    // Component of source vector along n-axis / w-axis
    // (Doubles as flux attenuation factor due to projection effects)
    double src_n = sqrt(1 - l * l - m * m);

    arma::vec src_offset = { l, m, src_n - 1 };
    src_offset *= -1;

    arma::cx_mat vis = flux * src_n * arma::exp(arma::cx_double(0.0, -2.0 * arma::datum::pi) * (dist_uvw * src_offset));

    return std::move(vis);
}

arma::cx_mat generate_visibilities_from_local_skymodel(arma::mat& skymodel, arma::mat& uvw_baselines)
{
    arma::cx_mat model_vis(uvw_baselines.n_rows, 1);
    model_vis.zeros();

    skymodel.each_row([&](arma::rowvec& src_entry) {
        model_vis += visibilities_for_point_source(uvw_baselines, src_entry[0], src_entry[1], src_entry[2]);
    });

    return std::move(model_vis);
}
}
