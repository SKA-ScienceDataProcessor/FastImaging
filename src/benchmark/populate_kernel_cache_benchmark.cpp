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

    for (auto _ : state) {
        arma::field<arma::Mat<cx_real_t>> cache;
        cache = stp::populate_kernel_cache(gaussiansinc, kernel_support, oversampling, pad, normalize);
    }
}

static void CustomArguments(benchmark::internal::Benchmark* b)
{
    for (int i = 2; i <= 16; i += 2)
        for (int j = 3; j <= 9; j += 2)
            b->Args({ i, j });
}

BENCHMARK(populate_kernel_cache_benchmark)
    ->Apply(CustomArguments);

BENCHMARK_MAIN();
