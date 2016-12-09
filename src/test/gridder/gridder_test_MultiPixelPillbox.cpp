#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <libstp.h>

class GridderMultiPixelPillbox : public ::testing::Test {
private:
    double v = 1. / 9.;

    int image_size;
    int support;
    double half_base_width;
    double oversampling;
    bool pad;
    bool normalize;

public:
    void
    SetUp()
    {
        image_size = 8;
        support = 1;
        half_base_width = 1.1;
        oversampling = oversampling_disabled;
        pad = false;
        normalize = false;

        vis = arma::ones<arma::cx_mat>(uv.n_rows);
    }

    void run()
    {
        result = convolve_to_grid(support, image_size, uv, vis, oversampling, pad, normalize, TopHat(half_base_width));
    }

    arma::mat uv = { { -2., 0 } };
    arma::cx_cube result;
    arma::cx_mat vis;
    arma::mat expected_result = {
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., v, v, v, 0., 0., 0., 0. },
        { 0., v, v, v, 0., 0., 0., 0. },
        { 0., v, v, v, 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. }
    };
};

TEST_F(GridderMultiPixelPillbox, equal)
{
    run();
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ accu(result.slice(VIS_GRID_INDEX)) }, arma::cx_mat{ accu(vis) }, "absdiff", tolerance));
}

TEST_F(GridderMultiPixelPillbox, vis_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_result, arma::real(result.slice(VIS_GRID_INDEX)), "absdiff", tolerance));
}

TEST_F(GridderMultiPixelPillbox, sampling_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_result, arma::real(result.slice(SAMPLING_GRID_INDEX)), "absdiff", tolerance));
}

TEST_F(GridderMultiPixelPillbox, GridderMultiPixelPillbox_benchmark)
{
    benchmark::RegisterBenchmark("MultiPixelPillbox", [this](benchmark::State& state) { while(state.KeepRunning())run(); });
    benchmark::RunSpecifiedBenchmarks();
}
