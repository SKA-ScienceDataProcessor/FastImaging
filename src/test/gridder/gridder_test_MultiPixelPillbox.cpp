#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class GridderMultiPixelPillbox : public ::testing::Test {
private:
    double v = 1. / 9.;
    int image_size;
    int support;
    double half_base_width;
    std::experimental::optional<int> oversampling;
    bool pad;
    bool normalize;

public:
    void
    SetUp()
    {
        image_size = 8;
        support = 1;
        half_base_width = 1.1;
        pad = false;
        normalize = true;
        vis = arma::ones<arma::cx_mat>(uv.n_rows);
    }

    void run()
    {
        result = convolve_to_grid(TopHat(half_base_width), support, image_size, uv, vis, oversampling, pad, normalize);
    }

    arma::mat uv = { { -2., 0 } };
    std::pair<arma::cx_mat, arma::cx_mat> result;
    arma::cx_mat vis;
    arma::mat expected_result = {
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., v, v, v, 0., 0., 0., 0. },
        { 0., v, v, v, 0., 0., 0., 0. },
        { 0., v, v, v, 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. }
    };
};

TEST_F(GridderMultiPixelPillbox, equal)
{
    run();
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ accu(std::get<vis_grid_index>(result)) }, arma::cx_mat{ accu(vis) }, "absdiff", tolerance));
}

TEST_F(GridderMultiPixelPillbox, vis_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_result, arma::real(std::get<vis_grid_index>(result)), "absdiff", tolerance));
}

TEST_F(GridderMultiPixelPillbox, sampling_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_result, arma::real(std::get<sampling_grid_index>(result)), "absdiff", tolerance));
}
