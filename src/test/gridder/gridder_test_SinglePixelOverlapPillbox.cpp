#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <libstp.h>

class GridderSinglePixelOverlapPillbox : public ::testing::Test {
private:
    int image_size;
    int support;
    double half_base_width;
    double vis_amplitude;
    bool pad;
    bool normalize;

public:
    void SetUp()
    {
        image_size = 8;
        support = 1;
        half_base_width = 0.5;
        vis_amplitude = 42.123;
        pad = false;
        normalize = false;

        vis = vis_amplitude * arma::ones<arma::cx_mat>(uv.n_cols);
    }

    void run()
    {
        result = convolve_to_grid(support, image_size, uv, vis, oversampling_disabled, pad, normalize, TopHat(half_base_width));
    }

    arma::mat uv = { { -2., 0 }, { -2, 0 } };
    arma::cx_mat vis;
    arma::cx_cube result;
    arma::mat expected_result = {
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 1., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. },
        { 0., 0., 0., 0., 0., 0., 0., 0. }
    };
};

TEST_F(GridderSinglePixelOverlapPillbox, equal)
{
    run();
    EXPECT_TRUE(arma::approx_equal(arma::cx_mat{ accu(arma::real(result.slice(VIS_GRID_INDEX))) }, arma::cx_mat{ accu(arma::real(vis)) }, "absdiff", tolerance));
}

TEST_F(GridderSinglePixelOverlapPillbox, vis_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal((expected_result * accu(arma::real(vis))), arma::real(result.slice(VIS_GRID_INDEX)), "absdiff", tolerance));
}

TEST_F(GridderSinglePixelOverlapPillbox, sampling_grid)
{
    run();
    EXPECT_TRUE(arma::approx_equal((expected_result * uv.n_cols), arma::real(result.slice(SAMPLING_GRID_INDEX)), "absdiff", tolerance));
}

TEST_F(GridderSinglePixelOverlapPillbox, GridderSinglePixelOverlapPillbox_benchmark)
{
    benchmark::RegisterBenchmark("SinglePixelOverlapPillbox", [this](benchmark::State& state) { while(state.KeepRunning())run(); });
    benchmark::RunSpecifiedBenchmarks();
}
