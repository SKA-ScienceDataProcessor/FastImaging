#include <gtest/gtest.h>
#include <libstp.h>

class GridderSinglePixelOverlapPillbox : public ::testing::Test {
public:
    void SetUp()
    {
        int n_image = 8;
        int support = 1;
        double half_base_width = 0.5;
        double vis_amplitude = 42.123;
        bool pad = false;
        bool normalize = false;

        vis = vis_amplitude * arma::ones<arma::cx_mat>(uv.n_cols);
        const TopHat top_hat(half_base_width);
        result = convolve_to_grid(support, n_image, uv, vis, NO_OVERSAMPLING, pad, normalize, top_hat);
    }

    arma::mat uv = { { -2., 0 }, { -2, 0 } };
    arma::cx_mat vis;
    arma::cx_cube result;
    arma::mat expected_result = {
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 1., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. }
    };
};

TEST_F(GridderSinglePixelOverlapPillbox, equal)
{
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ accu(arma::real(result.slice(VIS_GRID_INDEX))) }, arma::cx_mat{ accu(arma::real(vis)) }, "absdiff", tolerance));
}

TEST_F(GridderSinglePixelOverlapPillbox, vis_grid)
{
    EXPECT_TRUE(arma::approx_equal((expected_result * accu(arma::real(vis))), arma::real(result.slice(VIS_GRID_INDEX)), "absdiff", tolerance));
}

TEST_F(GridderSinglePixelOverlapPillbox, sampling_grid)
{
    EXPECT_TRUE(arma::approx_equal((expected_result * uv.n_cols), arma::real(result.slice(SAMPLING_GRID_INDEX)), "absdiff", tolerance));
}
