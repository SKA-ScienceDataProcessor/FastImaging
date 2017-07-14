/** @file gridder_test_benchmark.cpp
 *  @brief Test Gridder module performance
 *
 *  @bug No known bugs.
 */
#include <benchmark/benchmark.h>
#include <cblas.h>
#include <common/matrix_math.h>
#include <fixtures.h>
#include <stp.h>

#define DIVCONST 0.03549

std::vector<long> g_vsize = { 512, 1024, 1448, 2048, 2896, 4096, 5792, 8192, 11586, 16384, 23170, 32768 };

auto armadillo_accumulate_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    arma::Col<real_t> v = arma::vectorise(data);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(arma::accu(v));
    }
};

auto m_accumulate_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::mat_accumulate(data));
    }
};

auto m_accumulate_parallel_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::mat_accumulate_parallel(data));
    }
};

auto armadillo_mean_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    arma::Col<real_t> v = arma::vectorise(data);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(arma::mean(v));
    }
};

auto m_mean_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::mat_mean(data));
    }
};

auto m_mean_parallel_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::mat_mean_parallel(data));
    }
};

auto armadillo_stddev_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    arma::Col<real_t> v = arma::vectorise(data);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(arma::stddev(v));
    }
};

auto m_stddev_parallel_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::mat_stddev_parallel(data));
    }
};

auto m_mean_stddev_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::mat_mean_and_stddev(data));
    }
};

auto matrix_arma_inplace_div_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    double a = DIVCONST;

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(data /= a);
        benchmark::ClobberMemory();
    }
};

auto matrix_serial_inplace_div_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    double a = DIVCONST;

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(data.memptr());
        for (uint i = 0; i != data.n_elem; i++) {
            data.at(i) /= a;
        }
        benchmark::ClobberMemory();
    }
};

auto matrix_tbb_inplace_div_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    double a = DIVCONST;

    while (state.KeepRunning()) {
        uint n_elem = data.n_elem;
        benchmark::DoNotOptimize(data.memptr());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n_elem), [&](const tbb::blocked_range<size_t>& r) {
            for (uint i = r.begin(); i != r.end(); i++) {
                data[i] /= a;
            }
        });
        benchmark::ClobberMemory();
    }
};

auto matrix_cblas_inplace_div_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    double a = DIVCONST;

    while (state.KeepRunning()) {
        uint n_elem = data.n_elem;
        benchmark::DoNotOptimize(data.memptr());
#ifdef USE_FLOAT
        cblas_sscal(n_elem, (1.0f / a), reinterpret_cast<real_t*>(data.memptr()), 1); // division by a
#else
        cblas_dscal(n_elem, (1.0 / a), reinterpret_cast<real_t*>(data.memptr()), 1); // division by a
#endif
        benchmark::ClobberMemory();
    }
};

auto arma_shift_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> m = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    arma::Mat<real_t> out;

    int shift = std::copysign(1, state.range(1)) * g_vsize[state.range(1)];

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(out);
        out = arma::shift(m, shift, state.range(2));
        benchmark::ClobberMemory();
    }
};

auto matrix_shift_benchmark = [](benchmark::State& state) {
    long size = g_vsize[state.range(0)];
    arma::Mat<real_t> m = uncorrelated_gaussian_noise_background(size, size, 1.0, 0.0, 1);
    arma::Mat<real_t> out;

    int shift = std::copysign(1, state.range(1)) * g_vsize[state.range(1)];

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(out);
        out = stp::matrix_shift(m, shift, state.range(2));
        benchmark::ClobberMemory();
    }
};

int main(int argc, char** argv)
{
    benchmark::RegisterBenchmark("armadillo_accumulate_benchmark", armadillo_accumulate_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("stp_accumulate_benchmark", m_accumulate_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("stp_accumulate_parallel_benchmark", m_accumulate_parallel_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("armadillo_mean_benchmark", armadillo_mean_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("stp_mean_benchmark", m_mean_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("stp_mean_parallel_benchmark", m_mean_parallel_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("armadillo_stddev_benchmark", armadillo_stddev_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("stp_stddev_parallel_benchmark", m_stddev_parallel_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("stp_mean_stddev_benchmark", m_mean_stddev_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);

    // Inplace division

    benchmark::RegisterBenchmark("matrix_arma_inplace_div_benchmark", matrix_arma_inplace_div_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("matrix_serial_inplace_div_benchmark", matrix_serial_inplace_div_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("matrix_tbb_inplace_div_benchmark", matrix_tbb_inplace_div_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("matrix_cblas_inplace_div_benchmark", matrix_cblas_inplace_div_benchmark)
        ->DenseRange(1, 11)
        ->Unit(benchmark::kMicrosecond);

    // Matrix shift

    benchmark::RegisterBenchmark("arma_shift_benchmark", arma_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 1, 0, 0 })
        ->Args({ 1, 0, 1 });
    benchmark::RegisterBenchmark("matrix_shift_benchmark", matrix_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 1, 0, 0 })
        ->Args({ 1, 0, 1 });

    benchmark::RegisterBenchmark("arma_shift_benchmark", arma_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 2, 1, 0 })
        ->Args({ 2, 1, 1 });
    benchmark::RegisterBenchmark("matrix_shift_benchmark", matrix_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 2, 1, 0 })
        ->Args({ 2, 1, 1 });

    benchmark::RegisterBenchmark("arma_shift_benchmark", arma_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 3, 2, 0 })
        ->Args({ 3, 2, 1 });
    benchmark::RegisterBenchmark("matrix_shift_benchmark", matrix_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 3, 2, 0 })
        ->Args({ 3, 2, 1 });

    benchmark::RegisterBenchmark("arma_shift_benchmark", arma_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 4, 3, 0 })
        ->Args({ 4, 3, 1 });
    benchmark::RegisterBenchmark("matrix_shift_benchmark", matrix_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 4, 3, 0 })
        ->Args({ 4, 3, 1 });

    benchmark::RegisterBenchmark("arma_shift_benchmark", arma_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 5, 4, 0 })
        ->Args({ 5, 4, 1 });
    benchmark::RegisterBenchmark("matrix_shift_benchmark", matrix_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 5, 4, 0 })
        ->Args({ 5, 4, 1 });

    benchmark::RegisterBenchmark("arma_shift_benchmark", arma_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 6, 5, 0 })
        ->Args({ 6, 5, 1 });
    benchmark::RegisterBenchmark("matrix_shift_benchmark", matrix_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 6, 5, 0 })
        ->Args({ 6, 5, 1 });

    benchmark::RegisterBenchmark("arma_shift_benchmark", arma_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 7, 6, 0 })
        ->Args({ 7, 6, 1 });
    benchmark::RegisterBenchmark("matrix_shift_benchmark", matrix_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 7, 6, 0 })
        ->Args({ 7, 6, 1 });

    benchmark::RegisterBenchmark("arma_shift_benchmark", arma_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 8, 7, 0 })
        ->Args({ 8, 7, 1 });
    benchmark::RegisterBenchmark("matrix_shift_benchmark", matrix_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 8, 7, 0 })
        ->Args({ 8, 7, 1 });

    benchmark::RegisterBenchmark("arma_shift_benchmark", arma_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 9, 8, 0 })
        ->Args({ 9, 8, 1 });
    benchmark::RegisterBenchmark("matrix_shift_benchmark", matrix_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 9, 8, 0 })
        ->Args({ 9, 8, 1 });

    benchmark::RegisterBenchmark("arma_shift_benchmark", arma_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 10, 9, 0 })
        ->Args({ 10, 9, 1 });
    benchmark::RegisterBenchmark("matrix_shift_benchmark", matrix_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 10, 9, 0 })
        ->Args({ 10, 9, 1 });

    benchmark::RegisterBenchmark("arma_shift_benchmark", arma_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 11, 10, 0 })
        ->Args({ 11, 10, 1 });
    benchmark::RegisterBenchmark("matrix_shift_benchmark", matrix_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ 11, 10, 0 })
        ->Args({ 11, 10, 1 });

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
