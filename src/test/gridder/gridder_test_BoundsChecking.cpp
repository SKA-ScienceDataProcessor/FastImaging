#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <libstp.h>

void run()
{
    int image_size(8);
    int support(2);
    double half_base_width(1.5);
    double oversampling(oversampling_disabled);
    bool pad(false);
    bool normalize(false);

    arma::mat uv = { { -3., 0 } };
    arma::cx_mat vis = arma::ones<arma::cx_mat>(uv.n_rows);
    arma::cx_cube result = convolve_to_grid(support, image_size, uv, vis, oversampling, pad, normalize, TopHat(half_base_width));

    EXPECT_EQ(accu(arma::real(result.slice(VIS_GRID_INDEX))), 0);
}

TEST(GridderBoundsChecking, vis_grid)
{
    run();
}

// Benchmark functions

void BM_GridderBoundsChecking(benchmark::State& state)
{
    while (state.KeepRunning())
        run();
}

TEST(GridderBoundsChecking, vis_grid_benchmark)
{
    BENCHMARK(BM_GridderBoundsChecking);
    benchmark::RunSpecifiedBenchmarks();
}
