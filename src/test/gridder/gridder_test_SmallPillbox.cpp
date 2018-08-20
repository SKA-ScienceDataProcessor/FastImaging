#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class GridderSmallPillbox : public ::testing::Test {
private:
    real_t v = 1. / 4.;

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
        support = 1;
        half_base_width = 0.55;
        kernel_exact = true;
        oversampling = 1;
        shift_uv = false;
        halfplane_gridding = false;
        uv = { { -1.5, 0.5 } };
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
    arma::Mat<real_t> expected_result = {
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
    run();
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ arma::accu(arma::real(static_cast<arma::Mat<cx_real_t>>(result.vis_grid))) }, arma::cx_mat{ arma::accu(arma::real(expected_result)) }, "absdiff", fptolerance));
}

TEST_F(GridderSmallPillbox, vis_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_result, arma::real(static_cast<arma::Mat<cx_real_t>>(result.vis_grid)), "absdiff", fptolerance));
}
