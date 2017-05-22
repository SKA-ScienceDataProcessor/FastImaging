/** @file gridder_test_benchmark.cpp
 *  @brief Test Gridder module performance
 *
 *  @bug No known bugs.
 */
#include <benchmark/benchmark.h>
#include <stp.h>
#include <tbb/tbb.h>

auto malloc_standard_memset_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    cx_real_t* m = (cx_real_t*)std::malloc(sizeof(cx_real_t) * size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(m);
        std::memset(m, 0, sizeof(cx_real_t) * size * size);
        benchmark::ClobberMemory();
    }
    if (m)
        std::free(m);
};

auto malloc_parallel_memset_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    cx_real_t* m = (cx_real_t*)std::malloc(sizeof(cx_real_t) * size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(m);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, size * size), [&m](const tbb::blocked_range<size_t>& r) {
            std::memset(m + r.begin(), 0, sizeof(cx_real_t) * (r.end() - r.begin()));
        });
        benchmark::ClobberMemory();
    }
    if (m)
        std::free(m);
};

auto arma_standard_memset_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::Mat<cx_real_t> m(size, size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(m);
        std::memset(m.memptr(), 0, sizeof(cx_real_t) * size * size);
        benchmark::ClobberMemory();
    }
    m.set_size(0);
};

auto arma_parallel_memset_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
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

auto armadillo_fill_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::Mat<cx_real_t> m(size, size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(m);
        m.zeros();
        benchmark::ClobberMemory();
    }
    m.set_size(0);
};

auto armadillo_zero_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
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
    benchmark::RegisterBenchmark("malloc_standard_memset_benchmark", malloc_standard_memset_benchmark)
        ->Args({ 1 })
        ->Args({ 2 })
        ->Args({ 3 })
        ->Args({ 4 })
        ->Args({ 5 })
        ->Args({ 6 })
        ->Args({ 7 })
        ->Args({ 8 })
        ->Args({ 9 })
        ->Args({ 10 })
        ->Args({ 11 })
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("malloc_parallel_memset_benchmark", malloc_parallel_memset_benchmark)
        ->Args({ 1 })
        ->Args({ 2 })
        ->Args({ 3 })
        ->Args({ 4 })
        ->Args({ 5 })
        ->Args({ 6 })
        ->Args({ 7 })
        ->Args({ 8 })
        ->Args({ 9 })
        ->Args({ 10 })
        ->Args({ 11 })
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("arma_standard_memset_benchmark", malloc_standard_memset_benchmark)
        ->Args({ 1 })
        ->Args({ 2 })
        ->Args({ 3 })
        ->Args({ 4 })
        ->Args({ 5 })
        ->Args({ 6 })
        ->Args({ 7 })
        ->Args({ 8 })
        ->Args({ 9 })
        ->Args({ 10 })
        ->Args({ 11 })
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("arma_parallel_memset_benchmark", malloc_parallel_memset_benchmark)
        ->Args({ 1 })
        ->Args({ 2 })
        ->Args({ 3 })
        ->Args({ 4 })
        ->Args({ 5 })
        ->Args({ 6 })
        ->Args({ 7 })
        ->Args({ 8 })
        ->Args({ 9 })
        ->Args({ 10 })
        ->Args({ 11 })
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("armadillo_zero_benchmark", armadillo_zero_benchmark)
        ->Args({ 1 })
        ->Args({ 2 })
        ->Args({ 3 })
        ->Args({ 4 })
        ->Args({ 5 })
        ->Args({ 6 })
        ->Args({ 7 })
        ->Args({ 8 })
        ->Args({ 9 })
        ->Args({ 10 })
        ->Args({ 11 })
        ->Unit(benchmark::kMicrosecond);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
