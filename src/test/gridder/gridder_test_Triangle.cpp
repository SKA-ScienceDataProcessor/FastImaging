#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class GridderTriangle : public ::testing::Test {
private:
    const double _half_base_width = 2.0;
    const bool _pad = false;
    const bool _normalize = true;
    bool _kernel_exact_GRID = true;
    int _oversampling_GRID = 1;
    int _oversampling_KERNEL = 1;

public:
    void SetUp()
    {
        uv = { { 1.0, 0.0 } };
        vis = arma::ones<arma::cx_mat>(uv.n_rows);
        vis_weights = arma::ones<arma::mat>(uv.n_rows);
    }

    void run()
    {
        arma::mat subpix_offset = { 0.1, -0.15 };
        arma::mat uv_offset = uv + subpix_offset;

        // The last two parameters (false, false) force the use of the full gridder without uv shifting
        result = convolve_to_grid<true>(Triangle(_half_base_width), support, image_size, uv_offset, vis, vis_weights, _kernel_exact_GRID, _oversampling_GRID, _pad, _normalize, false, false);
        kernel = make_kernel_array(Triangle(_half_base_width), support, subpix_offset, _oversampling_KERNEL, _pad, _normalize);
    }

    const int image_size = 8;
    const int support = 2;
    GridderOutput result;
    arma::mat kernel;
    arma::cx_mat vis;
    arma::mat vis_weights;
    arma::mat uv;
};

TEST_F(GridderTriangle, equal)
{
    run();
    EXPECT_TRUE(std::abs(accu(arma::real(static_cast<arma::Mat<cx_real_t>>(result.vis_grid))) - accu(arma::real(vis))) < fptolerance);
}

TEST_F(GridderTriangle, uv_location)
{
    run();
    EXPECT_TRUE(arma::approx_equal(arma::real(static_cast<arma::Mat<cx_real_t>>(result.vis_grid)(arma::span(image_size / 2 - support, image_size / 2 + support), arma::span(image_size / 2 + 1 - support, image_size / 2 + 1 + support))), arma::conv_to<arma::Mat<real_t>>::from(kernel / accu(kernel)), "absdiff", fptolerance));
}
