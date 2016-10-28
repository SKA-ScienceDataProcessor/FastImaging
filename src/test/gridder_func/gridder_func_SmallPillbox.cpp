#include "../../libstp/gridder/gridder.h"
#include "gtest/gtest.h"

class GridderSmallPillbox : public ::testing::Test {
    int n_image = 8;
    int support = 1;
    double half_base_width = 0.55;
    double oversampling = NO_OVERSAMPLING;
    bool pad = false;
    bool normalize = false;
    bool raise_bounds = true;
    arma::mat uv = {{-1.5, 0.5}};
    double v = 1. / 4.;

    public:
        arma::cx_mat vis = arma::ones<arma::cx_mat>(uv.n_rows);
        arma::cx_cube result = convolve_to_grid<TopHat>(support, n_image, uv, vis, oversampling, pad, normalize, raise_bounds, half_base_width);
        arma::mat expected_result =
            {
                { 0., 0., 0., 0., 0., 0., 0., 0. },
                { 0., 0., 0., 0., 0., 0., 0., 0. },
                { 0., 0., 0., 0., 0., 0., 0., 0. },
                { 0., 0., 0., 0., 0., 0., 0., 0. },
                { 0., 0.,  v,  v, 0., 0., 0., 0. },
                { 0., 0.,  v,  v, 0., 0., 0., 0. },
                { 0., 0., 0., 0., 0., 0., 0., 0. },
                { 0., 0., 0., 0., 0., 0., 0., 0. }
            };
};

TEST_F(GridderSmallPillbox, equal) {
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{accu(arma::real(result.slice(VIS_GRID_INDEX)))}, arma::cx_mat{accu(arma::real(vis))}, "absdiff", tolerance));
}

TEST_F(GridderSmallPillbox, vis_grid) {
    EXPECT_TRUE(arma::approx_equal(expected_result, arma::real(result.slice(VIS_GRID_INDEX)), "absdiff", tolerance));
}
