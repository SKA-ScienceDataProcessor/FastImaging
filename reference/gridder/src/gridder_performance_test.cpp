#include <benchmark/benchmark.h>
#include <cppGridderPrototype.h> // Original code had no header

static void BM_performFFT(benchmark::State& state)
{
    while(state.KeepRunning() == true) {
        ; // TODO
    }
}

BENCHMARK(BM_performFFT);

BENCHMARK_MAIN();
