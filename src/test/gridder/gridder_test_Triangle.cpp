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
    }

    void run()
    {
        arma::mat subpix_offset = { 0.1, -0.15 };
        arma::mat uv_offset = uv + subpix_offset;

        result = convolve_to_grid(Triangle(_half_base_width), support, image_size, uv_offset, vis, _kernel_exact_GRID, _oversampling_GRID, _pad, _normalize);
        kernel = make_kernel_array(Triangle(_half_base_width), support, subpix_offset, _oversampling_KERNEL, _pad, _normalize);
    }

    const int image_size = 8;
    const int support = 2;
    std::pair<arma::Mat<cx_real_t>, arma::Mat<real_t> > result;
    arma::mat kernel;
    arma::cx_mat vis;
    arma::mat uv;
};

TEST_F(GridderTriangle, equal)
{
    run();
    EXPECT_TRUE(std::abs(accu(arma::real(std::get<vis_grid_index>(result))) - accu(arma::real(vis))) < tolerance);
}

TEST_F(GridderTriangle, uv_location)
{
    run();
    EXPECT_TRUE(arma::approx_equal(arma::real(std::get<vis_grid_index>(result)(arma::span(image_size / 2 - support, image_size / 2 + support), arma::span(image_size / 2 + 1 - support, image_size / 2 + 1 + support))), arma::conv_to<arma::Mat<real_t> >::from(kernel / accu(kernel)), "absdiff", tolerance));
}
