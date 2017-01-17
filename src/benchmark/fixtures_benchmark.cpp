/** @file fixtures_test_benchmark.cpp
 *  @brief Test Fixtures module performance
 *
 *  @bug No known bugs.
 */
#include <benchmark/benchmark.h>
#include <stp.h>

void run()
{
}

static void fixtures_test_benchmark(benchmark::State& state)
{
    while (state.KeepRunning())
        run();
}

BENCHMARK(fixtures_test_benchmark);
BENCHMARK_MAIN()
