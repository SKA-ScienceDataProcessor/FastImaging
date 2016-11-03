#include "../../libstp/gridder/gridder.h"
#include "gtest/gtest.h"

class GridderNearbyComplexVis : public::testing::Test {
    int n_image = 8;
    int support = 2;
    double half_base_width = 1.1;
    double oversampling = NO_OVERSAMPLING;
    bool pad = false;
    bool normalize = false;
    bool raise_bounds = true;
    arma::mat uv = {{-2., 1}, {0, -1}};
    std::complex<double> v =  1. / 9. + 0i;

    public:
        arma::cx_mat vis = arma::ones<arma::cx_mat>(uv.n_rows);
        arma::cx_cube result = convolve_to_grid<TopHat>(support, n_image, uv, vis, oversampling, pad, normalize, raise_bounds, half_base_width);
        arma::cx_mat expected_vis_grid =
            {
                { 0.+0i, 0.+0i, 0.+0i, 0.+0i, 0.+0i, 0.+0i, 0.+0i, 0.+0i },
                { 0.+0i, 0.+0i, 0.+0i, 0.+0i, 0.+0i, 0.+0i, 0.+0i, 0.+0i },
                { 0.+0i, 0.+0i, 0.+0i,     v,     v,     v, 0.+0i, 0.+0i },
                { 0.+0i, 0.+0i, 0.+0i,     v,     v,     v, 0.+0i, 0.+0i },
                { 0.+0i,     v,     v,  2.*v,     v,     v, 0.+0i, 0.+0i },
                { 0.+0i,     v,     v,     v, 0.+0i, 0.+0i, 0.+0i, 0.+0i },
                { 0.+0i,     v,     v,     v, 0.+0i, 0.+0i, 0.+0i, 0.+0i },
                { 0.+0i, 0.+0i, 0.+0i, 0.+0i, 0.+0i, 0.+0i, 0.+0i, 0.+0i }
            };
};

TEST_F(GridderNearbyComplexVis, equal) {
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{accu(result.slice(VIS_GRID_INDEX))}, arma::cx_mat{accu(vis)}, "absdiff", tolerance));
}

TEST_F(GridderNearbyComplexVis, vis_grid) {
    EXPECT_TRUE(arma::approx_equal(expected_vis_grid, result.slice(VIS_GRID_INDEX), "absdiff", tolerance));
}

TEST_F(GridderNearbyComplexVis, sampling_grid) {
    EXPECT_TRUE(arma::approx_equal(expected_vis_grid, result.slice(SAMPLING_GRID_INDEX), "absdiff", tolerance));
}
