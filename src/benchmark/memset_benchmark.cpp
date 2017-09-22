/** @file memset_benchmark.cpp
 *  @brief Test memset performance
 */
#include <benchmark/benchmark.h>
#include <stp.h>
#include <tbb/tbb.h>

auto standard_memset_benchmark = [](benchmark::State& state) {
    size_t size = pow(2, double(state.range(0)));
    arma::Mat<cx_real_t> m(size, size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(m);
        std::memset(m.memptr(), 0, sizeof(cx_real_t) * size * size);
        benchmark::ClobberMemory();
    }
    m.set_size(0);
};

auto parallel_memset_benchmark = [](benchmark::State& state) {
    size_t size = pow(2, double(state.range(0)));
    arma::Mat<cx_real_t> m(size, size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(m);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, size * size), [&m](const tbb::blocked_range<size_t>& r) {
            std::memset(m.memptr() + r.begin(), 0, sizeof(cx_real_t) * (r.end() - r.begin()));
        });
        benchmark::ClobberMemory();
    }
    m.set_size(0);
};

auto armadillo_zero_benchmark = [](benchmark::State& state) {
    size_t size = pow(2, double(state.range(0)));
    arma::Mat<cx_real_t> m(size, size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(m);
        m.zeros();
        benchmark::ClobberMemory();
    }
    m.set_size(0);
};

int main(int argc, char** argv)
{
    benchmark::RegisterBenchmark("standard_memset_benchmark", standard_memset_benchmark)
        ->DenseRange(10, 16)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("parallel_memset_benchmark", parallel_memset_benchmark)
        ->DenseRange(10, 16)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("armadillo_zero_benchmark", armadillo_zero_benchmark)
        ->DenseRange(10, 16)
        ->Unit(benchmark::kMicrosecond);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
