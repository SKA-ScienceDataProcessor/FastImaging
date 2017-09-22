#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

/**
 * Test halfplane convolve_to_grid with shifting
 *
 */

TEST(GridderOversampledGridding, equal)
{
    real_t v = 1. / 9.;
    int image_size = 8;
    int support = 1;
    int oversampling = 9;
    bool kernel_exact = false;
    double half_base_width = 1.1;

    arma::mat uv = {
        { -2.0, 0.0 }, // Should be replicated in (2.0, 0.0)
        { 0.0, -2.0 }, // Not replicated. Simply converted to (0.0, 2.0)
    };

    arma::cx_mat vis;
    arma::mat vis_weights;
    vis = arma::ones<arma::cx_mat>(uv.n_rows, 1);
    vis_weights = arma::ones<arma::mat>(uv.n_rows, 1);
    arma::Mat<real_t> expected_result = {
        { 0., v, v, v, 0., v, v, v },
        { 0., v, v, 2 * v, v, 2 * v, v, v },
        { 0., 0., 0., v, v, v, 0., 0. },
        { 0., 0., 0., v, v, v, 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
    };

    GridderOutput shifted_res = convolve_to_grid<true>(TopHat(half_base_width), support, image_size, uv, vis, vis_weights, kernel_exact, oversampling);
    arma::Mat<cx_real_t> result_image = matrix_shift(shifted_res.sampling_grid, image_size / 2, 1);
    arma::Mat<cx_real_t> result_beam = matrix_shift(shifted_res.vis_grid, image_size / 2, 1);

    kernel_exact = true;
    GridderOutput shifted_res_exact = convolve_to_grid<true>(TopHat(half_base_width), support, image_size, uv, vis, vis_weights, kernel_exact, oversampling);
    arma::Mat<cx_real_t> result_image_exact = matrix_shift(shifted_res_exact.sampling_grid, image_size / 2, 1);
    arma::Mat<cx_real_t> result_beam_exact = matrix_shift(shifted_res_exact.vis_grid, image_size / 2, 1);

    // Get halfplane vis: needed for testing
    convert_to_halfplane_visibilities(uv, vis, vis_weights, support);

    EXPECT_TRUE(arma::accu(arma::real(result_image)) - arma::accu(arma::real(vis)) < fptolerance);
    EXPECT_NEAR(arma::accu(arma::real(result_image) - arma::real(result_image_exact)), 0.0, fptolerance);
    EXPECT_NEAR(arma::accu(arma::real(result_beam) - arma::real(result_beam_exact)), 0.0, fptolerance);
    EXPECT_NEAR(arma::accu(arma::real(result_beam) - expected_result), 0.0, fptolerance);
}
