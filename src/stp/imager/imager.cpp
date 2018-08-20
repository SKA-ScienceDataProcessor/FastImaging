/**
 * @file imager.cpp
 * @brief Implementation of the imager functions.
 */

#include "imager.h"

namespace stp {

std::vector<std::chrono::high_resolution_clock::time_point> times_iv;

ImageVisibilities::ImageVisibilities(
    const arma::cx_mat& vis,
    const arma::mat& vis_weights,
    const arma::mat& uvw_lambda,
    const ImagerPars& img_pars = ImagerPars(),
    const W_ProjectionPars& w_proj = W_ProjectionPars(),
    const A_ProjectionPars& a_proj = A_ProjectionPars())
{
    std::pair<arma::Mat<real_t>, arma::Mat<real_t>> result;

    // Since "image_visibilities" is a function template, there's no way to prevent the use of the following switch statement.
    // Inheritance could be used instead, but virtual function calls would impose an unnecessary performance penalty.
    switch (img_pars.kernel_function) {
    case stp::KernelFunction::TopHat: {
        stp::TopHat kernel_function(img_pars.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(vis), std::move(vis_weights),
            std::move(uvw_lambda), img_pars, w_proj, a_proj);
    };
        break;
    case stp::KernelFunction::Triangle: {
        stp::Triangle kernel_function(img_pars.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(vis), std::move(vis_weights),
            std::move(uvw_lambda), img_pars, w_proj, a_proj);
    };
        break;
    case stp::KernelFunction::Sinc: {
        stp::Sinc kernel_function(img_pars.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(vis), std::move(vis_weights),
            std::move(uvw_lambda), img_pars, w_proj, a_proj);
    };
        break;
    case stp::KernelFunction::Gaussian: {
        stp::Gaussian kernel_function(img_pars.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(vis), std::move(vis_weights),
            std::move(uvw_lambda), img_pars, w_proj, a_proj);
    };
        break;
    case stp::KernelFunction::GaussianSinc: {
        stp::GaussianSinc kernel_function(img_pars.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(vis), std::move(vis_weights),
            std::move(uvw_lambda), img_pars, w_proj, a_proj);
    };
        break;
    case stp::KernelFunction::PSWF: {
        stp::PSWF kernel_function(img_pars.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(vis), std::move(vis_weights),
            std::move(uvw_lambda), img_pars, w_proj, a_proj);
    };
        break;
    default:
        assert(0);
        break;
    }

    vis_grid = std::move(result.first);
    sampling_grid = std::move(result.second);
}
}
