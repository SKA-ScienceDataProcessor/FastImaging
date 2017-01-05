/** @file visibility_test_benchmark.cpp
 *  @brief Test Visibility module performance
 *
 *  @bug No known bugs.
 */

#include <benchmark/benchmark.h>
#include <stp.h>

void run()
{
}

static void visibility_test_benchmark(benchmark::State& state)
{
    while (state.KeepRunning())
        run();
}

BENCHMARK(visibility_test_benchmark);
BENCHMARK_MAIN()
