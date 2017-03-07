/** @file kernel_test_benchmark.cpp
 *  @brief Test kernel module performance
 *
 *  @bug No known bugs.
 */

#include <benchmark/benchmark.h>
#include <stp.h>

static void kernel_test_benchmark(benchmark::State& state)
{
    arma::mat offset_index = { 0., 0. };
    int support = state.range(0);
    int oversampling = 1;

    while (state.KeepRunning()) {
        // Use GaussianSinc kernel
        stp::GaussianSinc kernel_creator(support);
        benchmark::DoNotOptimize(make_kernel_array(kernel_creator, support, offset_index, oversampling));
        benchmark::ClobberMemory();
    }
}

BENCHMARK(kernel_test_benchmark)
    ->Args({ 3 })
    ->Args({ 5 })
    ->Args({ 7 });
BENCHMARK_MAIN()
