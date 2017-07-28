#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

/**
 * Integration test of the convolve_to_grid function with oversampling
 *
 *  Mostly tests same functionality as 'test_kernel_caching'.
 *
 */

TEST(GridderOversampledGridding, equal)
{
    int image_size = 8;
    int support = 2;
    int oversampling = 9;
    bool kernel_exact = false;

    arma::mat uv = {
        { 1.0, 0.0 },
        { 1.3, 0.0 },
        { 0.01, -1.32 },
    };

    arma::cx_mat vis;
    arma::mat vis_weights;
    vis = arma::ones<arma::cx_mat>(uv.n_rows, 1);
    vis_weights = arma::ones<arma::mat>(uv.n_rows, 1);

    // Let's grid a triangle function
    Triangle triangle(2.0);
    GridderOutput result = convolve_to_grid<true>(triangle, support, image_size, uv, vis, vis_weights, kernel_exact, oversampling, false, true, false, false);

    EXPECT_TRUE(arma::accu(arma::real(static_cast<arma::Mat<cx_real_t>>(result.sampling_grid))) - arma::accu(arma::real(vis)) < tolerance);
}
