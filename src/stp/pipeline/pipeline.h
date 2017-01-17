#ifndef PIPELINE_H
#define PIPELINE_H

/**
* @file pipeline.h
* Contains the prototypes and implementation of pipeline function
*/

#include "../convolution/conv_func.h"
#include "../imager/imager.h"
#include "../sourcefind/sourcefind.h"
#include "../visibility/visibility.h"
#include <utility>

namespace stp {

/**
 * @brief generate_pipeline function
 *
 * This function (template + functor) generate a pipeline source_find_image
 *
 * @param[in] uvw_lambda (arma::cx_mat): The array of complex visibilities.
 * @param[in] vis_noise_level (double): This defines the std. dev./sigma of the Gaussian distribution.
 * @param[in] image_size (int): Width of the image in pixels
 * @param[in] cell_size (double): Angular-width of a synthesized pixel in the image to be created
 * @param[in] detection_n_sigma (double): Detection threshold as multiple of RMS
 * @param[in] analysis_n_sigma (double): Analysis threshold as multiple of RMS
 * @param[in] kernel_support (double): Controls kernel-generation
 *
 * @return result of the pipeline on the uvw_lambda inputed
 */
template <typename T>
source_find_image generate_pipeline(
    arma::mat uvw_lambda,
    double vis_noise_level,
    int image_size,
    double cell_size,
    double detection_n_sigma,
    double analysis_n_sigma,
    double kernel_support,
    const T& kernel_creator)
{
    SkyCoord pointing_centre(180, 8);
    SkyRegion field_of_view(pointing_centre, 1);
    SkyCoord northeast_of_centre(pointing_centre._ra + 0.01, pointing_centre._dec + 0.01);

    std::vector<SkySource> steady_source_list;
    steady_source_list.push_back(SkySource(pointing_centre, 1));
    steady_source_list.push_back(SkySource(northeast_of_centre, 0.4));

    SkyCoord southwest_of_centre(field_of_view._centre._ra - 0.05, field_of_view._centre._dec - 0.05);
    std::vector<SkySource> transient_src_list;
    transient_src_list.push_back(SkySource(southwest_of_centre, 0.5));

    std::vector<SkySource> source_list_w_transient;
    source_list_w_transient.reserve(steady_source_list.size() + transient_src_list.size());
    source_list_w_transient.insert(source_list_w_transient.end(), steady_source_list.begin(), steady_source_list.end());
    source_list_w_transient.insert(source_list_w_transient.end(), transient_src_list.begin(), transient_src_list.end());

    arma::cx_mat data_vis = visibilities_for_source_list(pointing_centre, source_list_w_transient, uvw_lambda);
    data_vis = add_gaussian_noise(vis_noise_level, data_vis);

    arma::cx_mat model_vis = visibilities_for_source_list(pointing_centre, steady_source_list, uvw_lambda);

    arma::cx_mat residual_vis = data_vis - model_vis;
    std::pair<arma::cx_mat, arma::cx_mat> result = image_visibilities(kernel_creator, residual_vis, uvw_lambda, image_size, cell_size, kernel_support, true, 0);

    return source_find_image(arma::real(result.first), detection_n_sigma, analysis_n_sigma, 1, true);
}
}

#endif /* PIPELINE_H */
