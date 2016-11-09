#include <libstp.h>
#include <gtest/gtest.h>

class GridderMultiPixelPillbox : public::testing::Test {
    int n_image = 8;
    int support = 1;
    double half_base_width = 1.1;
    double oversampling = NO_OVERSAMPLING;
    bool pad = false;
    bool normalize = false;
    arma::mat uv = {{-2., 0}};
    double v = 1. / 9.;

    public:
        arma::cx_mat vis = arma::ones<arma::cx_mat>(uv.n_rows);
        arma::cx_cube result = convolve_to_grid<TopHat>(support, n_image, uv, vis, oversampling, pad, normalize, half_base_width);
        arma::mat expected_result =
            {
                { 0., 0., 0., 0., 0., 0., 0., 0. },
                { 0., 0., 0., 0., 0., 0., 0., 0. },
                { 0., 0., 0., 0., 0., 0., 0., 0. },
                { 0.,  v,  v,  v, 0., 0., 0., 0. },
                { 0.,  v,  v,  v, 0., 0., 0., 0. },
                { 0.,  v,  v,  v, 0., 0., 0., 0. },
                { 0., 0., 0., 0., 0., 0., 0., 0. },
                { 0., 0., 0., 0., 0., 0., 0., 0. }
            };
};

TEST_F(GridderMultiPixelPillbox, equal) {    
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{accu(result.slice(VIS_GRID_INDEX))}, arma::cx_mat{accu(vis)}, "absdiff", tolerance));
}

TEST_F(GridderMultiPixelPillbox, vis_grid) {
    EXPECT_TRUE(arma::approx_equal(expected_result, arma::real(result.slice(VIS_GRID_INDEX)), "absdiff", tolerance));
}

TEST_F(GridderMultiPixelPillbox, sampling_grid) {
    EXPECT_TRUE(arma::approx_equal(expected_result, arma::real(result.slice(SAMPLING_GRID_INDEX)), "absdiff", tolerance));
}
