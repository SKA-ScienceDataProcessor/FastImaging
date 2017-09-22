/** @file fftshift_benchmark.cpp
 *  @brief Test fft shift performance
 */

#include <../stp/common/fft.h>
#include <benchmark/benchmark.h>
#include <fixtures.h>
#include <stp.h>

static void fftshift_benchmark(benchmark::State& state)
{
    size_t size = pow(2, state.range(0));
    arma::Mat<real_t> m = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(m);
        stp::fftshift(m);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(fftshift_benchmark)
    ->DenseRange(10, 16)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN()
