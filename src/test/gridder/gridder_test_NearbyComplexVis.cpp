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
    bool pad;
    bool normalize;

public:
    void SetUp()
    {
        image_size = 8;
        support = 2;
        half_base_width = 1.1;
        kernel_exact = true;
        oversampling = 1;
        pad = false;
        normalize = true;
        uv = { { -2., 1 }, { 0, -1 } };

        vis = arma::ones<arma::cx_mat>(uv.n_rows);
    }

    void run()
    {
        // The last two parameters (false, false) force the use of the full gridder without uv shifting
        result = convolve_to_grid(TopHat(half_base_width), support, image_size, uv, vis, kernel_exact, oversampling, pad, normalize, false, false);
    }

    arma::mat uv;
    arma::cx_mat vis;
    std::pair<MatStp<cx_real_t>, MatStp<cx_real_t>> result;
    arma::Mat<cx_real_t> expected_vis_grid = {
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, v, v, v, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, v, v, v, 0.0, 0.0 },
        { 0.0, v, v, real_t(2.) * v, v, v, 0.0, 0.0 },
        { 0.0, v, v, v, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, v, v, v, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
    };
};

TEST_F(GridderNearbyComplexVis, equal)
{
    run();
    EXPECT_TRUE(std::abs(accu(static_cast<arma::Mat<cx_real_t>>(std::get<vis_grid_index>(result))) - cx_real_t(accu(vis))) < tolerance);
}

TEST_F(GridderNearbyComplexVis, vis_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_vis_grid, static_cast<arma::Mat<cx_real_t>>(std::get<vis_grid_index>(result)), "absdiff", tolerance));
}

TEST_F(GridderNearbyComplexVis, sampling_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_vis_grid, static_cast<arma::Mat<cx_real_t>>(std::get<sampling_grid_index>(result)), "absdiff", tolerance));
}
