#include <gtest/gtest.h>
#include <libstp.h>

class GridderNearbyComplexVis : public ::testing::Test {
private:
    std::complex<double> v = 1. / 9.;

public:
    void SetUp()
    {
        int n_image = 8;
        int support = 2;
        double half_base_width = 1.1;
        bool pad = false;
        bool normalize = false;
        arma::mat uv = { { -2., 1 }, { 0, -1 } };

        vis = arma::ones<arma::cx_mat>(uv.n_rows);
        const TopHat top_hat(half_base_width);
        result = convolve_to_grid(support, n_image, uv, vis, NO_OVERSAMPLING, pad, normalize, top_hat);
    }

    arma::cx_mat vis;
    arma::cx_cube result;
    arma::cx_mat expected_vis_grid = {
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, v, v, v, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, v, v, v, 0.0, 0.0 },
        { 0.0, v, v, 2. * v, v, v, 0.0, 0.0 },
        { 0.0, v, v, v, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, v, v, v, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
    };
};

TEST_F(GridderNearbyComplexVis, equal)
{
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ accu(result.slice(VIS_GRID_INDEX)) }, arma::cx_mat{ accu(vis) }, "absdiff", tolerance));
}

TEST_F(GridderNearbyComplexVis, vis_grid)
{
    EXPECT_TRUE(arma::approx_equal(expected_vis_grid, result.slice(VIS_GRID_INDEX), "absdiff", tolerance));
}

TEST_F(GridderNearbyComplexVis, sampling_grid)
{
    EXPECT_TRUE(arma::approx_equal(expected_vis_grid, result.slice(SAMPLING_GRID_INDEX), "absdiff", tolerance));
}
