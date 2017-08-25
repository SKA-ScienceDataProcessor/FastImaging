/** @file median_benchmark.cpp
 *  @brief Test median functions performance
 */

#include <benchmark/benchmark.h>
#include <cblas.h>
#include <common/matrix_math.h>
#include <fixtures.h>
#include <stp.h>

std::vector<long> g_vsize = { 1024, 1025, 2048, 2049, 4096, 4097, 8192, 8193, 16384, 16385, 32768, 32769 };

auto armadillo_median_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    arma::Col<real_t> v = arma::vectorise(data);
    state.SetLabel(std::to_string(size));
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(arma::median(v));
    }
};

auto stp_median_exact_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    state.SetLabel(std::to_string(size));

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::mat_median_exact(data));
    }
};

auto stp_binmedian_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    state.SetLabel(std::to_string(size));

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::mat_binmedian(data));
    }
};

auto stp_binapprox_median_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    state.SetLabel(std::to_string(size));

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::mat_median_binapprox(data));
    }
};

int main(int argc, char** argv)
{
    benchmark::RegisterBenchmark("armadillo_median_benchmark", armadillo_median_benchmark)
        ->DenseRange(0, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("stp_median_exact_benchmark", stp_median_exact_benchmark)
        ->DenseRange(0, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("stp_binmedian_benchmark", stp_binmedian_benchmark)
        ->DenseRange(0, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("stp_binapprox_median_benchmark", stp_binapprox_median_benchmark)
        ->DenseRange(0, 11)
        ->Unit(benchmark::kMicrosecond);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
