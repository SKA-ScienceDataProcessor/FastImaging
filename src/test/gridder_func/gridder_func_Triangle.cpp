#include <gtest/gtest.h>
#include <libstp.h>

class GridderTriangle : public ::testing::Test {
private:
    const double _half_base_width = 2.0;
    const double _triangle_value = 1.0;

    const double _oversampling_GRID = NO_OVERSAMPLING;
    const double _oversampling_KERNEL = 1.0;
    const bool _pad = false;
    const bool _normalize = false;

public:
    void SetUp()
    {
        arma::mat uv_1 = { { 1.0, 0.0 } };
        arma::mat subpix_offset = { 0.1, -0.15 };
        vis = arma::ones<arma::cx_mat>(uv_1.n_rows);
        arma::mat uv_2 = uv_1 + subpix_offset;

        const Triangle triangle(_half_base_width, _triangle_value);
        result = convolve_to_grid(support, n_image, uv_2, vis, _oversampling_GRID, _pad, _normalize, triangle);
        kernel = make_kernel_array(support, subpix_offset, _oversampling_KERNEL, _pad, _normalize, triangle);
    }

    const int n_image = 8;
    const int support = 2;
    arma::cx_cube result;
    arma::mat kernel;
    arma::cx_mat vis;
};

TEST_F(GridderTriangle, equal)
{
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ accu(arma::real(result.slice(VIS_GRID_INDEX))) }, arma::cx_mat{ accu(arma::real(vis)) }, "absdiff", tolerance));
}

TEST_F(GridderTriangle, uv_location)
{
    arma::span xrange = arma::span(n_image / 2 + 1 - support, n_image / 2 + 1 + support);
    arma::span yrange = arma::span(n_image / 2 - support, n_image / 2 + support);
    EXPECT_TRUE(arma::approx_equal(arma::real(result.slice(VIS_GRID_INDEX)(yrange, xrange)), (kernel / accu(kernel)), "absdiff", tolerance));
}
