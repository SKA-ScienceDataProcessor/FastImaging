#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class GridderMultipleComplexVis : public ::testing::Test {
private:
    cx_real_t v = 1. / 9.;

    int image_size;
    int support;
    double half_base_width;
    bool kernel_exact;
    int oversampling;
    bool pad;
    bool normalize;

public:
    void SetUp()
    {
        image_size = 8;
        support = 1;
        half_base_width = 1.1;
        kernel_exact = true;
        oversampling = 1;
        pad = false;
        normalize = true;
        uv = { { -2., 1 }, { 1, -1 } };

        vis = arma::ones<arma::cx_mat>(uv.n_rows);
        vis_weights = arma::ones<arma::mat>(uv.n_rows);
    }

    void run()
    {
        // The last two parameters (false, false) force the use of the full gridder without uv shifting
        GridderOutput res = convolve_to_grid<true>(TopHat(half_base_width), support, image_size, uv, vis, vis_weights, kernel_exact, oversampling, pad, normalize, false, false);
        result = std::make_pair(static_cast<arma::Mat<cx_real_t>>(res.vis_grid), static_cast<arma::Mat<cx_real_t>>(res.sampling_grid));
    }

    arma::mat uv;
    std::pair<arma::Mat<cx_real_t>, arma::Mat<cx_real_t>> result;
    arma::cx_mat vis;
    arma::mat vis_weights;
    arma::Mat<cx_real_t> expected_result = {
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, v, v, v, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, v, v, v, 0.0 },
        { 0.0, v, v, v, v, v, v, 0.0 },
        { 0.0, v, v, v, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, v, v, v, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
    };
};

TEST_F(GridderMultipleComplexVis, equal)
{
    run();
    EXPECT_TRUE(arma::approx_equal(arma::Mat<cx_real_t>{ accu(result.first) }, arma::Mat<cx_real_t>{ cx_real_t(accu(vis)) }, "absdiff", tolerance));
}

TEST_F(GridderMultipleComplexVis, vis_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_result, result.first, "absdiff", tolerance));
}

TEST_F(GridderMultipleComplexVis, sampling_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_result, result.second, "absdiff", tolerance));
}
