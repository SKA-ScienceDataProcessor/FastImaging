#include <libstp.h>
#include <gtest/gtest.h>

class GridderTriangle : public ::testing::Test {
    double half_base_width = 2.0;
    double oversampling_GRID = NO_OVERSAMPLING;
    double oversampling_KERNEL = 1.0;
    double oversampling = 1.0;
    bool pad = false;
    bool normalize = false;
    double triangle_value = 1.0;
    
    public: 
        int n_image = 8;
        int support = 2;
        arma::mat uv_1 = {{1.0, 0.0}};
        arma::mat subpix_offset = {0.1, -0.15};
        arma::cx_mat vis = arma::ones<arma::cx_mat>(uv_1.n_rows);
        arma::mat uv_2 = uv_1 + subpix_offset; 
        arma::cx_cube result = convolve_to_grid<Triangle>(support, n_image, uv_2, vis, oversampling_GRID, pad, normalize, half_base_width, triangle_value);
        arma::mat kernel = make_kernel_array<Triangle>(support, subpix_offset, oversampling_KERNEL, pad, normalize, half_base_width, triangle_value);
};

TEST_F(GridderTriangle, equal) {
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{accu(arma::real(result.slice(VIS_GRID_INDEX)))}, arma::cx_mat{accu(arma::real(vis))}, "absdiff", tolerance));
}

TEST_F(GridderTriangle, uv_location) {
    arma::span xrange = arma::span(n_image / 2 + 1 - support, n_image / 2 + 1 + support);
    arma::span yrange = arma::span(n_image / 2 - support, n_image / 2 + support);
    EXPECT_TRUE(arma::approx_equal( arma::real(result.slice(VIS_GRID_INDEX)(yrange, xrange)), (kernel/accu(kernel)), "absdiff", tolerance));
}
