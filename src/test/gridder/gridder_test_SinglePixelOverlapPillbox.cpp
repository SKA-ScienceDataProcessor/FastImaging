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
    bool shift_uv;
    bool halfplane_gridding;

public:
    void SetUp()
    {
        image_size = 8;
        support = 1;
        half_base_width = 0.5;
        vis_amplitude = 42.123;
        kernel_exact = true;
        oversampling = 1;
        shift_uv = false;
        halfplane_gridding = false;

        vis = vis_amplitude * arma::ones<arma::cx_mat>(uv.n_cols);
        vis_weights = arma::ones<arma::mat>(uv.n_cols);
    }

    void run()
    {
        // The last two parameters (false, false) force the use of the full gridder without uv shifting
        GridderOutput res = convolve_to_grid<true>(TopHat(half_base_width), support, image_size, uv, vis, vis_weights, kernel_exact, oversampling, shift_uv, halfplane_gridding);
        result = std::make_pair(static_cast<arma::Mat<cx_real_t>>(res.vis_grid), static_cast<arma::Mat<cx_real_t>>(res.sampling_grid));
    }

    arma::mat uv = { { -2., 0 }, { -2, 0 } };
    arma::cx_mat vis;
    arma::mat vis_weights;
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
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ accu(arma::real(result.first)) }, arma::cx_mat{ accu(arma::real(vis)) }, "absdiff", fptolerance));
}

TEST_F(GridderSinglePixelOverlapPillbox, vis_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal((expected_result * accu(arma::real(vis))), arma::real(result.first), "absdiff", fptolerance));
}

TEST_F(GridderSinglePixelOverlapPillbox, sampling_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal((expected_result * uv.n_cols), arma::real(result.second), "absdiff", fptolerance));
}
