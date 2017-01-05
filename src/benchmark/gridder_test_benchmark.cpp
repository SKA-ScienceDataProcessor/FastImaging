/** @file gridder_test_benchmark.cpp
 *  @brief Test Gridder module performance
 *
 *  @bug No known bugs.
 */
#include <benchmark/benchmark.h>
#include <stp.h>

void run()
{
}

static void gridder_test_benchmark(benchmark::State& state)
{
    while (state.KeepRunning())
        run();
}

BENCHMARK(gridder_test_benchmark);
BENCHMARK_MAIN()
