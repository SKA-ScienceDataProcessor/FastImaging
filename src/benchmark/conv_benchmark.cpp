/** @file conv_test_benchmark.cpp
 *  @brief Test Convolution module performance
 *
 *  @bug No known bugs.
 */

#include <benchmark/benchmark.h>
#include <stp.h>

void run()
{
}

static void conv_test_benchmark(benchmark::State& state)
{
    while (state.KeepRunning())
        run();
}

BENCHMARK(conv_test_benchmark);
BENCHMARK_MAIN()
