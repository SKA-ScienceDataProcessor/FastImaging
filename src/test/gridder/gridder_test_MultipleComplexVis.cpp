#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class GridderMultipleComplexVis : public ::testing::Test {
private:
    std::complex<double> v = 1. / 9.;

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
        support = 2;
        half_base_width = 1.1;
        kernel_exact = true;
        oversampling = 1;
        pad = false;
        normalize = true;
        uv = { { -2., 1 }, { 1, -1 } };

        vis = arma::ones<arma::cx_mat>(uv.n_rows);
    }

    void run()
    {
        result = convolve_to_grid(TopHat(half_base_width), support, image_size, uv, vis, kernel_exact, oversampling, pad, normalize);
    }

    arma::mat uv;
    std::pair<arma::cx_mat, arma::mat> result;
    arma::cx_mat vis;
    arma::cx_mat expected_result = {
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
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ accu(std::get<vis_grid_index>(result)) }, arma::cx_mat{ accu(vis) }, "absdiff", tolerance));
}

TEST_F(GridderMultipleComplexVis, vis_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_result, std::get<vis_grid_index>(result), "absdiff", tolerance));
}

TEST_F(GridderMultipleComplexVis, sampling_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(arma::real(expected_result), std::get<sampling_grid_index>(result), "absdiff", tolerance));
}
