#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

TEST(GridderBoundsChecking, vis_grid)
{
    int image_size(8);
    int support(2);
    double half_base_width(1.5);
    std::experimental::optional<int> oversampling;
    bool pad(false);
    bool normalize(true);

    arma::mat uv = { { -3., 0 } };
    arma::cx_mat vis = arma::ones<arma::cx_mat>(uv.n_rows);
    std::pair<arma::cx_mat, arma::cx_mat> result = convolve_to_grid(TopHat(half_base_width), support, image_size, uv, vis, oversampling, pad, normalize);

    EXPECT_EQ(accu(arma::real(std::get<vis_grid_index>(result))), 0);
}
