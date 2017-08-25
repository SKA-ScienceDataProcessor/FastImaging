/** @file populate_kernel_cache_benchmark.cpp
 *  @brief Test populate kernel cache module performance
 */
#include <benchmark/benchmark.h>
#include <stp.h>

static void populate_kernel_cache_benchmark(benchmark::State& state)
{
    bool pad = false;
    bool normalize = true;
    stp::GaussianSinc gaussiansinc;
    int oversampling = state.range(0);
    int kernel_support = state.range(1);

    while (state.KeepRunning())
        benchmark::DoNotOptimize(stp::populate_kernel_cache(gaussiansinc, kernel_support, oversampling, pad, normalize));
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
