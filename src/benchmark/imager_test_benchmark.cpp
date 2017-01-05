/** @file imager_test_benchmark.cpp
 *  @brief Test Imager module performance
 *
 *  @bug No known bugs.
 */

#include <benchmark/benchmark.h>
#include <stp.h>

void run()
{
}

static void imager_test_benchmark(benchmark::State& state)
{
    while (state.KeepRunning())
        run();
}

BENCHMARK(imager_test_benchmark);
BENCHMARK_MAIN()
