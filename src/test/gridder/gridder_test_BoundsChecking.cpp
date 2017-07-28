#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

TEST(GridderBoundsChecking, vis_grid)
{
    int image_size = 8;
    int support = 2;
    double half_base_width = 1.5;
    bool kernel_exact = true;
    int oversampling = 1;
    bool pad = false;
    bool normalize = true;

    arma::mat bad_uv = { { -3., 0 } };
    arma::cx_mat vis = arma::ones<arma::cx_mat>(bad_uv.n_rows);
    arma::mat vis_weights = arma::ones<arma::mat>(bad_uv.n_rows);
    GridderOutput grid = convolve_to_grid<true>(TopHat(half_base_width), support, image_size, bad_uv, vis, vis_weights, kernel_exact, oversampling, pad, normalize);
    arma::Mat<cx_real_t>& vis_grid = grid.vis_grid;
    arma::Mat<cx_real_t>& sampling_grid = grid.sampling_grid;

    EXPECT_EQ(accu(arma::real(vis_grid)), 0);
    EXPECT_EQ(accu(arma::real(sampling_grid)), 0);

    // Now check we're filtering indices in the correct order
    // The mixed
    arma::mat good_uv = { { 0, 0 } };
    arma::mat mixed_uv = { { -3., 0 }, { 0, 0 } };
    vis = arma::ones<arma::cx_mat>(good_uv.n_rows);
    vis_weights = arma::ones<arma::mat>(good_uv.n_rows);
    GridderOutput good_grid = convolve_to_grid<true>(TopHat(half_base_width), support, image_size, good_uv, vis, vis_weights, kernel_exact, oversampling, pad, normalize);
    arma::Mat<cx_real_t>& good_vis_grid = good_grid.vis_grid;
    arma::Mat<cx_real_t>& good_sampling_grid = good_grid.sampling_grid;

    vis = arma::ones<arma::cx_mat>(mixed_uv.n_rows);
    vis_weights = arma::ones<arma::mat>(mixed_uv.n_rows);
    GridderOutput mixed_grid = convolve_to_grid<true>(TopHat(half_base_width), support, image_size, mixed_uv, vis, vis_weights, kernel_exact, oversampling, pad, normalize);
    arma::Mat<cx_real_t>& mixed_vis_grid = mixed_grid.vis_grid;
    arma::Mat<cx_real_t>& mixed_sampling_grid = mixed_grid.sampling_grid;

    EXPECT_EQ(accu(arma::real(good_vis_grid - mixed_vis_grid)), 0);
    EXPECT_EQ(accu(arma::real(good_sampling_grid - mixed_sampling_grid)), 0);
}
