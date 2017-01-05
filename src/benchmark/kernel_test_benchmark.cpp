/** @file kernel_test_benchmark.cpp
 *  @brief Test kernel module performance
 *
 *  @bug No known bugs.
 */

#include <benchmark/benchmark.h>
#include <stp.h>

void run()
{
}

static void kernel_test_benchmark(benchmark::State& state)
{
    while (state.KeepRunning())
        run();
}

BENCHMARK(kernel_test_benchmark);
BENCHMARK_MAIN()
