#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <libstp.h>

class GridderNearbyComplexVis : public ::testing::Test {
private:
    std::complex<double> v = 1. / 9.;

    int image_size;
    int support;
    double half_base_width;
    bool pad;
    bool normalize;

public:
    void SetUp()
    {
        image_size = 8;
        support = 2;
        half_base_width = 1.1;
        pad = false;
        normalize = false;
        uv = { { -2., 1 }, { 0, -1 } };

        vis = arma::ones<arma::cx_mat>(uv.n_rows);
    }

    void run()
    {
        result = convolve_to_grid(support, image_size, uv, vis, oversampling_disabled, pad, normalize, TopHat(half_base_width));
    }

    arma::mat uv;
    arma::cx_mat vis;
    arma::cx_cube result;
    arma::cx_mat expected_vis_grid = {
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, v, v, v, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, v, v, v, 0.0, 0.0 },
        { 0.0, v, v, 2. * v, v, v, 0.0, 0.0 },
        { 0.0, v, v, v, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, v, v, v, 0.0, 0.0, 0.0, 0.0 },
        { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }
    };
};

TEST_F(GridderNearbyComplexVis, equal)
{
    run();
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ accu(result.slice(VIS_GRID_INDEX)) }, arma::cx_mat{ accu(vis) }, "absdiff", tolerance));
}

TEST_F(GridderNearbyComplexVis, vis_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_vis_grid, result.slice(VIS_GRID_INDEX), "absdiff", tolerance));
}

TEST_F(GridderNearbyComplexVis, sampling_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal(expected_vis_grid, result.slice(SAMPLING_GRID_INDEX), "absdiff", tolerance));
}

TEST_F(GridderNearbyComplexVis, GridderNearbyComplexVis_benchmark)
{
    benchmark::RegisterBenchmark("NearbyComplexVis", [this](benchmark::State& state) { while(state.KeepRunning())run(); });
    benchmark::RunSpecifiedBenchmarks();
}
