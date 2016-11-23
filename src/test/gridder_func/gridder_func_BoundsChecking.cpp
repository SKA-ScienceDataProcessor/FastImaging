#include <gtest/gtest.h>
#include <libstp.h>

TEST(GridderBoundsChecking, vis_grid)
{
    int n_image(8);
    int support(2);
    double half_base_width(1.5);
    double oversampling(NO_OVERSAMPLING);
    bool pad(false);
    bool normalize(false);

    arma::mat uv = { { -3., 0 } };
    arma::cx_mat vis = arma::ones<arma::cx_mat>(uv.n_rows);
    arma::cx_cube result = convolve_to_grid<TopHat>(support, n_image, uv, vis, oversampling, pad, normalize, half_base_width);

    EXPECT_EQ(accu(arma::real(result.slice(VIS_GRID_INDEX))), 0);
}
