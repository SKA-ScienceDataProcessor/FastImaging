/** @file gridder_test_benchmark.cpp
 *  @brief Test Gridder module performance
 *
 *  @bug No known bugs.
 */
#include <benchmark/benchmark.h>
#include <cblas.h>
#include <common/vector_math.h>
#include <stp.h>

#define DIVCONST 0.03549

arma::vec generate_vector_data(int size)
{
    arma::arma_rng::set_seed(1);
    return std::move(arma::randu<arma::vec>(size) + 10);
}

arma::cx_vec generate_cx_vector_data(int size)
{
    arma::arma_rng::set_seed(1);
    return std::move(arma::randu<arma::cx_vec>(size) + 10);
}

auto standard_memset_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    std::complex<double>* m = (std::complex<double>*)std::malloc(sizeof(std::complex<double>) * size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(m);
        std::memset(m, 0, sizeof(std::complex<double>) * size * size);
        benchmark::ClobberMemory();
    }
    if (m)
        std::free(m);
};

auto armadillo_memset_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::cx_mat m(size, size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(m);
        m.zeros();
        benchmark::ClobberMemory();
    }
};

auto armadillo_accumulate_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(arma::accu(v));
    }
};

auto vector_accumulate_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_accumulate(v));
    }
};

auto vector_accumulate_parallel_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_accumulate_parallel(v));
    }
};

auto armadillo_mean_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(arma::mean(v));
    }
};

auto vector_mean_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_mean(v));
    }
};

auto vector_mean_parallel_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_mean_parallel(v));
    }
};

auto vector_mean_robust_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_mean_robust(v));
    }
};

auto vector_mean_robust_parallel_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_mean_parallel(v));
    }
};

auto armadillo_stddev_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(arma::stddev(v));
    }
};

auto vector_stddev_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_stddev(v));
    }
};

auto vector_stddev_parallel_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_stddev_parallel(v));
    }
};

auto vector_stddev_robust_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_stddev_robust(v));
    }
};

auto vector_stddev_robust_parallel_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::vec v = generate_vector_data(size * size);

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_stddev_parallel(v));
    }
};

auto matrix_arma_inplace_div_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::cx_vec cv = generate_cx_vector_data(size * size);
    double a = DIVCONST;

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(cv /= a);
        benchmark::ClobberMemory();
    }
};

auto matrix_serial_inplace_div_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::cx_vec cv = generate_cx_vector_data(size * size);
    double a = DIVCONST;

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(cv.memptr());
        for (uint i = 0; i != cv.n_elem; i++) {
            cv.at(i) /= a;
        }
        benchmark::ClobberMemory();
    }
};

auto matrix_tbb_inplace_div_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::cx_vec cv = generate_cx_vector_data(size * size);
    double a = DIVCONST;

    while (state.KeepRunning()) {
        uint n_elem = cv.n_elem;
        benchmark::DoNotOptimize(cv.memptr());
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n_elem), [&cv, &a](const tbb::blocked_range<size_t>& r) {
            for (uint i = r.begin(); i != r.end(); i++) {
                cv[i] /= a;
            }
        });
        benchmark::ClobberMemory();
    }
};

auto matrix_cblas_inplace_div_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::cx_vec cv = generate_cx_vector_data(size * size);
    double a = DIVCONST;

    while (state.KeepRunning()) {
        uint n_elem = cv.n_elem;
        benchmark::DoNotOptimize(cv.memptr());
        cblas_zdscal(n_elem, (1.0 / a), cv.memptr(), 1); // for division we use (1 / a)
        benchmark::ClobberMemory();
    }
};

auto arma_shift_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::arma_rng::set_seed(1);
    arma::cx_mat m = arma::randu<arma::cx_mat>(size, size);
    arma::cx_mat out;

    int shift = std::copysign(1, state.range(1)) * pow(2, std::abs(double(state.range(1) + 19) / 2.0));

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(out);
        out = arma::shift(m, shift, state.range(2));
        benchmark::ClobberMemory();
    }
};

auto matrix_shift_benchmark = [](benchmark::State& state) {
    int size = pow(2, double(state.range(0) + 19) / 2.0);
    arma::arma_rng::set_seed(1);
    arma::cx_mat m = arma::randu<arma::cx_mat>(size, size);
    arma::cx_mat out;

    int shift = std::copysign(1, state.range(1)) * pow(2, std::abs(double(state.range(1) + 19) / 2.0));

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(out);
        out = stp::matrix_shift(m, shift, state.range(2));
        benchmark::ClobberMemory();
    }
};

int main(int argc, char** argv)
{

    benchmark::RegisterBenchmark("standard_memset_benchmark", standard_memset_benchmark)
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
    benchmark::RegisterBenchmark("armadillo_memset_benchmark", armadillo_memset_benchmark)
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
    benchmark::RegisterBenchmark("armadillo_accumulate_benchmark", armadillo_accumulate_benchmark)
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
    benchmark::RegisterBenchmark("vector_accumulate_benchmark", vector_accumulate_benchmark)
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
    benchmark::RegisterBenchmark("vector_accumulate_parallel_benchmark", vector_accumulate_parallel_benchmark)
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
    benchmark::RegisterBenchmark("armadillo_mean_benchmark", armadillo_mean_benchmark)
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
    benchmark::RegisterBenchmark("vector_mean_benchmark", vector_mean_benchmark)
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
    benchmark::RegisterBenchmark("vector_mean_parallel_benchmark", vector_mean_parallel_benchmark)
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
    benchmark::RegisterBenchmark("vector_mean_robust_benchmark", vector_mean_robust_benchmark)
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
    benchmark::RegisterBenchmark("vector_mean_robust_parallel_benchmark", vector_mean_robust_parallel_benchmark)
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
    benchmark::RegisterBenchmark("armadillo_stddev_benchmark", armadillo_stddev_benchmark)
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
    benchmark::RegisterBenchmark("vector_stddev_benchmark", vector_stddev_benchmark)
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
    benchmark::RegisterBenchmark("vector_stddev_parallel_benchmark", vector_stddev_parallel_benchmark)
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
    benchmark::RegisterBenchmark("vector_stddev_robust_benchmark", vector_stddev_robust_benchmark)
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
    benchmark::RegisterBenchmark("vector_stddev_robust_parallel_benchmark", vector_stddev_robust_parallel_benchmark)
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

    /////////////////////

    benchmark::RegisterBenchmark("matrix_arma_inplace_div_benchmark", matrix_arma_inplace_div_benchmark)
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
    benchmark::RegisterBenchmark("matrix_serial_inplace_div_benchmark", matrix_serial_inplace_div_benchmark)
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
    benchmark::RegisterBenchmark("matrix_tbb_inplace_div_benchmark", matrix_tbb_inplace_div_benchmark)
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
    benchmark::RegisterBenchmark("matrix_cblas_inplace_div_benchmark", matrix_cblas_inplace_div_benchmark)
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
