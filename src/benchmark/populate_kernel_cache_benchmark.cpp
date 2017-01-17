/** @file gridder_test_benchmark.cpp
 *  @brief Test Gridder module performance
 *
 *  @bug No known bugs.
 */
#include <benchmark/benchmark.h>
#include <stp.h>

bool pad = false;
bool normalize = true;
stp::GaussianSinc gaussiansinc;

void run(int oversampling_cache = 9, int support = 3)
{
    arma::field<arma::mat> kernel_cache = stp::populate_kernel_cache(gaussiansinc, support, oversampling_cache, pad, normalize);
}

static void populate_kernel_cache_benchmark(benchmark::State& state)
{
    while (state.KeepRunning())
        run(state.range(0), state.range(1));
}

BENCHMARK(populate_kernel_cache_benchmark)
    ->Args({ 3, 3 })
    ->Args({ 5, 3 })
    ->Args({ 7, 3 })
    ->Args({ 9, 3 })
    ->Args({ 11, 3 })
    ->Args({ 13, 3 })
    ->Args({ 15, 3 })
    ->Args({ 3, 5 })
    ->Args({ 5, 5 })
    ->Args({ 7, 5 })
    ->Args({ 9, 5 })
    ->Args({ 11, 5 })
    ->Args({ 13, 5 })
    ->Args({ 15, 5 })
    ->Args({ 3, 7 })
    ->Args({ 5, 7 })
    ->Args({ 7, 7 })
    ->Args({ 9, 7 })
    ->Args({ 11, 7 })
    ->Args({ 13, 7 })
    ->Args({ 15, 7 });

BENCHMARK_MAIN()
