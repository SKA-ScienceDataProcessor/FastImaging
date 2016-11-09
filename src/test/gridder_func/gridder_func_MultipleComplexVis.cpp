#include <libstp.h>
#include <gtest/gtest.h>

class GridderMultipleComplexVis : public::testing::Test {
    int n_image = 8;
    int support = 2;
    double half_base_width = 1.1;
    double oversampling = NO_OVERSAMPLING;
    bool pad = false;
    bool normalize = false;
    arma::mat uv = {{-2., 1}, {1, -1}};
    std::complex<double> com_one;
    std::complex<double> v = 1./9.;
    
    public:
        arma::cx_mat vis = arma::ones<arma::cx_mat>(uv.n_rows);
        arma::cx_cube result = convolve_to_grid<TopHat>(support, n_image, uv, vis, oversampling, pad, normalize, half_base_width);
        arma::cx_mat expected_result =
            {
                { com_one, com_one, com_one, com_one, com_one, com_one, com_one, com_one },
                { com_one, com_one, com_one, com_one, com_one, com_one, com_one, com_one },
                { com_one, com_one, com_one, com_one,       v,       v,       v, com_one },
                { com_one, com_one, com_one, com_one,       v,       v,       v, com_one },
                { com_one,       v,       v,       v,       v,       v,       v, com_one },
                { com_one,       v,       v,       v, com_one, com_one, com_one, com_one },
                { com_one,       v,       v,       v, com_one, com_one, com_one, com_one },
                { com_one, com_one, com_one, com_one, com_one, com_one, com_one, com_one }
            };
};

TEST_F(GridderMultipleComplexVis, equal) {
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{accu(result.slice(VIS_GRID_INDEX))}, arma::cx_mat{accu(vis)}, "absdiff", tolerance));
}

TEST_F(GridderMultipleComplexVis, vis_grid) {
    EXPECT_TRUE(arma::approx_equal(expected_result, result.slice(VIS_GRID_INDEX), "absdiff", tolerance));
}

TEST_F(GridderMultipleComplexVis, sampling_grid) {
    EXPECT_TRUE(arma::approx_equal(expected_result, result.slice(SAMPLING_GRID_INDEX), "absdiff", tolerance));
}
