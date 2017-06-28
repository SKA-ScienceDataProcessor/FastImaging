#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class GridderSinglePixelOverlapPillbox : public ::testing::Test {
private:
    int image_size;
    int support;
    double half_base_width;
    double vis_amplitude;
    bool kernel_exact;
    int oversampling;
    bool pad;
    bool normalize;

public:
    void SetUp()
    {
        image_size = 8;
        support = 1;
        half_base_width = 0.5;
        vis_amplitude = 42.123;
        kernel_exact = true;
        oversampling = 1;
        pad = false;
        normalize = false;

        vis = vis_amplitude * arma::ones<arma::cx_mat>(uv.n_cols);
    }

    void run()
    {
        // The last two parameters (false, false) force the use of the full gridder without uv shifting
        result = convolve_to_grid(TopHat(half_base_width), support, image_size, uv, vis, kernel_exact, oversampling, pad, normalize, false, false);
    }

    arma::mat uv = { { -2., 0 }, { -2, 0 } };
    arma::cx_mat vis;
    std::pair<arma::Mat<cx_real_t>, arma::Mat<cx_real_t>> result;
    arma::Mat<real_t> expected_result = {
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
    run();
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ accu(arma::real(std::get<vis_grid_index>(result))) }, arma::cx_mat{ accu(arma::real(vis)) }, "absdiff", tolerance));
}

TEST_F(GridderSinglePixelOverlapPillbox, vis_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal((expected_result * accu(arma::real(vis))), arma::real(std::get<vis_grid_index>(result)), "absdiff", tolerance));
}

TEST_F(GridderSinglePixelOverlapPillbox, sampling_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal((expected_result * uv.n_cols), arma::real(std::get<sampling_grid_index>(result)), "absdiff", tolerance));
}
