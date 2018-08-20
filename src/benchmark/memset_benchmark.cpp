/** @file memset_benchmark.cpp
 *  @brief Test memset performance
 */
#include <benchmark/benchmark.h>
#include <stp.h>
#include <tbb/tbb.h>

auto standard_memset_benchmark = [](benchmark::State& state) {
    size_t size = state.range(0);
    arma::Mat<cx_real_t> m(size, size);

    for (auto _ : state) {
        benchmark::DoNotOptimize(m.memptr());
        std::memset(m.memptr(), 0, sizeof(cx_real_t) * size * size);
        benchmark::ClobberMemory();
    }
    m.set_size(0);
};

auto parallel_memset_benchmark = [](benchmark::State& state) {
    size_t size = state.range(0);
    arma::Mat<cx_real_t> m(size, size);

    for (auto _ : state) {
        benchmark::DoNotOptimize(m.memptr());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, size * size), [&m](const tbb::blocked_range<size_t>& r) {
            std::memset(m.memptr() + r.begin(), 0, sizeof(cx_real_t) * (r.end() - r.begin()));
        });
        benchmark::ClobberMemory();
    }
    m.set_size(0);
};

auto armadillo_zero_benchmark = [](benchmark::State& state) {
    size_t size = state.range(0);
    arma::Mat<cx_real_t> m(size, size);

    for (auto _ : state) {
        benchmark::DoNotOptimize(m.memptr());
        m.zeros();
        benchmark::ClobberMemory();
    }
    m.set_size(0);
};

int main(int argc, char** argv)
{
    benchmark::RegisterBenchmark("standard_memset_benchmark", standard_memset_benchmark)
        ->RangeMultiplier(2)
        ->Range(1 << 10, 1 << 16)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("parallel_memset_benchmark", parallel_memset_benchmark)
        ->RangeMultiplier(2)
        ->Range(1 << 10, 1 << 16)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("armadillo_zero_benchmark", armadillo_zero_benchmark)
        ->RangeMultiplier(2)
        ->Range(1 << 10, 1 << 16)
        ->Unit(benchmark::kMicrosecond);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
