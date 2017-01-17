/** @file sourcefind_test_benchmark.cpp
 *  @brief Test SourceFindImage module performance
 *
 *  @bug No known bugs.
 */

#include <benchmark/benchmark.h>
#include <stp.h>

void run()
{
}

static void sourcefind_test_benchmark(benchmark::State& state)
{
    while (state.KeepRunning())
        run();
}

BENCHMARK(sourcefind_test_benchmark);
BENCHMARK_MAIN()
