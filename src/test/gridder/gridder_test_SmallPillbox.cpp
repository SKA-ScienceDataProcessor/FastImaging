#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <libstp.h>

class GridderSmallPillbox : public ::testing::Test {
private:
    double v = 1. / 4.;

    int image_size;
    int support;
    double half_base_width;
    double oversampling;
    bool pad;
    bool normalize;

public:
    void SetUp()
    {
        image_size = 8;
        support = 1;
        half_base_width = 0.55;
        oversampling = oversampling_disabled;
        pad = false;
        normalize = false;
        uv = { { -1.5, 0.5 } };
        vis = arma::ones<arma::cx_mat>(uv.n_rows);
    }

    void run()
    {
        result = convolve_to_grid(support, image_size, uv, vis, oversampling, pad, normalize, TopHat(half_base_width));
    }

    arma::mat uv;
    arma::cx_mat vis;
    arma::cx_cube result;
    arma::mat expected_result = {
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., v, v, 0., 0., 0., 0. },
        { 0., 0., v, v, 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. }
    };
};

TEST_F(GridderSmallPillbox, equal)
{
    run();
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ arma::accu(arma::real(result.slice(VIS_GRID_INDEX))) }, arma::cx_mat{ arma::accu(arma::real(expected_result)) }, "absdiff", tolerance));
}

TEST_F(GridderSmallPillbox, vis_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_result, arma::real(result.slice(VIS_GRID_INDEX)), "absdiff", tolerance));
}

TEST_F(GridderSmallPillbox, GridderSmallPillbox_benchmark)
{
    benchmark::RegisterBenchmark("SmallPillbox", [this](benchmark::State& state) { while(state.KeepRunning())run(); });
    benchmark::RunSpecifiedBenchmarks();
}
