#include <gtest/gtest.h>
#include <libstp.h>

class GridderSmallPillbox : public ::testing::Test {
private:
    double v = 1. / 4.;

public:
    void SetUp()
    {
        int n_image = 8;
        int support = 1;
        double half_base_width = 0.55;
        double oversampling = NO_OVERSAMPLING;
        bool pad = false;
        bool normalize = false;
        arma::mat uv = { { -1.5, 0.5 } };

        arma::cx_mat vis = arma::ones<arma::cx_mat>(uv.n_rows);
        const TopHat top_hat(half_base_width);
        result = convolve_to_grid(support, n_image, uv, vis, oversampling, pad, normalize, top_hat);
    }

    arma::cx_mat vis;
    arma::cx_cube result;
    arma::mat expected_result = {
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., v, v, 0., 0., 0., 0. },
        { 0., 0., v, v, 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. }
    };
};

TEST_F(GridderSmallPillbox, equal)
{
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ arma::accu(arma::real(result.slice(VIS_GRID_INDEX))) }, arma::cx_mat{ arma::accu(arma::real(expected_result)) }, "absdiff", tolerance));
}

TEST_F(GridderSmallPillbox, vis_grid)
{
    EXPECT_TRUE(arma::approx_equal(expected_result, arma::real(result.slice(VIS_GRID_INDEX)), "absdiff", tolerance));
}
