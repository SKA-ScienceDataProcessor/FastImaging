/** @file fftshift_benchmark.cpp
 *  @brief Test fft shift performance
 */

#include <../stp/common/fft.h>
#include <benchmark/benchmark.h>
#include <fixtures.h>
#include <stp.h>

static void fftshift_benchmark(benchmark::State& state)
{
    size_t size = state.range(0);
    arma::Mat<real_t> m = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(m.memptr());
        stp::fftshift(m);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(fftshift_benchmark)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 16)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
