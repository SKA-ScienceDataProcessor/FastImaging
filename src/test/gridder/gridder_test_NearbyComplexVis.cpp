#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class GridderNearbyComplexVis : public ::testing::Test {
private:
    cx_real_t v = 1. / 9.;

    int image_size;
    int support;
    double half_base_width;
    bool kernel_exact;
    int oversampling;
    bool shift_uv;
    bool halfplane_gridding;

public:
    void SetUp()
    {
        image_size = 8;
        support = 2;
        half_base_width = 1.1;
        kernel_exact = true;
        oversampling = 1;
        shift_uv = false;
        halfplane_gridding = false;
        uv = { { -1., 1 }, { 0, -1 } };

        vis = arma::ones<arma::cx_mat>(uv.n_rows);
        vis_weights = arma::ones<arma::mat>(uv.n_rows);
    }

    void run()
    {
        // The last two parameters (false, false) force the use of the full gridder without uv shifting
        result = convolve_to_grid<true>(TopHat(half_base_width), support, image_size, uv, vis, vis_weights, kernel_exact, oversampling, shift_uv, halfplane_gridding);
    }

    arma::mat uv;
    arma::cx_mat vis;
    arma::mat vis_weights;
    GridderOutput result;
    arma::Mat<cx_real_t> expected_vis_grid = {
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, v, v, v, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, v, v, v, 0.0, 0.0 },
        { 0.0, 0.0, v, real_t(2.) * v, real_t(2.) * v, v, 0.0, 0.0 },
        { 0.0, 0.0, v, v, v, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, v, v, v, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
    };
};

TEST_F(GridderNearbyComplexVis, equal)
{
    run();
    EXPECT_TRUE(std::abs(accu(static_cast<arma::Mat<cx_real_t>>(result.vis_grid)) - cx_real_t(accu(vis))) < fptolerance);
}

TEST_F(GridderNearbyComplexVis, vis_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_vis_grid, static_cast<arma::Mat<cx_real_t>>(result.vis_grid), "absdiff", fptolerance));
}

TEST_F(GridderNearbyComplexVis, sampling_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_vis_grid, static_cast<arma::Mat<cx_real_t>>(result.sampling_grid), "absdiff", fptolerance));
}
