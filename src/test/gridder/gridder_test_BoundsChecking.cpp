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
    std::pair<arma::cx_mat, arma::cx_mat> grid = convolve_to_grid(TopHat(half_base_width), support, image_size, bad_uv, vis, kernel_exact, oversampling, pad, normalize);

    EXPECT_EQ(accu(arma::real(std::get<vis_grid_index>(grid))), 0);
    EXPECT_EQ(accu(arma::real(std::get<sampling_grid_index>(grid))), 0);

    // Now check we're filtering indices in the correct order
    // The mixed
    arma::mat good_uv = { { 0, 0 } };
    arma::mat mixed_uv = { { -3., 0 }, { 0, 0 } };
    vis = arma::ones<arma::cx_mat>(good_uv.n_rows);
    std::pair<arma::cx_mat, arma::cx_mat> good_grid = convolve_to_grid(TopHat(half_base_width), support, image_size, good_uv, vis, kernel_exact, oversampling, pad, normalize);

    vis = arma::ones<arma::cx_mat>(mixed_uv.n_rows);
    std::pair<arma::cx_mat, arma::cx_mat> mixed_grid = convolve_to_grid(TopHat(half_base_width), support, image_size, mixed_uv, vis, kernel_exact, oversampling, pad, normalize);

    EXPECT_EQ(accu(arma::real(std::get<vis_grid_index>(good_grid) - std::get<vis_grid_index>(mixed_grid))), 0);
    EXPECT_EQ(accu(arma::real(std::get<sampling_grid_index>(good_grid) - std::get<sampling_grid_index>(mixed_grid))), 0);
}
