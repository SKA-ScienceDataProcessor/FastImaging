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
    vis = arma::ones<arma::cx_mat>(uv.n_rows, 1);
    arma::Mat<real_t> expected_result = {
        { 0., v, v, v, 0., v, v, v },
        { 0., v, v, 2 * v, v, 2 * v, v, v },
        { 0., 0., 0., v, v, v, 0., 0. },
        { 0., 0., 0., v, v, v, 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
    };

    // Convert visibilities to single halfplane
    convert_to_halfplane_visibilities(uv, vis, support);

    std::pair<MatStp<cx_real_t>, MatStp<cx_real_t>> shifted_res = convolve_to_grid(TopHat(half_base_width), support, image_size, uv, vis, kernel_exact, oversampling);
    arma::Mat<cx_real_t> result_image = matrix_shift(shifted_res.first, image_size / 2, 1);
    arma::Mat<cx_real_t> result_beam = matrix_shift(shifted_res.second, image_size / 2, 1);

    kernel_exact = 1;
    std::pair<MatStp<cx_real_t>, MatStp<cx_real_t>> shifted_res_exact = convolve_to_grid(TopHat(half_base_width), support, image_size, uv, vis, kernel_exact, oversampling);
    arma::Mat<cx_real_t> result_image_exact = matrix_shift(shifted_res_exact.first, image_size / 2, 1);
    arma::Mat<cx_real_t> result_beam_exact = matrix_shift(shifted_res_exact.second, image_size / 2, 1);

    EXPECT_TRUE(arma::accu(arma::real(result_image)) - arma::accu(arma::real(vis)) < tolerance);
    EXPECT_NEAR(arma::accu(arma::real(result_image) - arma::real(result_image_exact)), 0.0, tolerance);
    EXPECT_NEAR(arma::accu(arma::real(result_beam) - arma::real(result_beam_exact)), 0.0, tolerance);
    EXPECT_NEAR(arma::accu(arma::real(result_beam) - expected_result), 0.0, tolerance);
}
