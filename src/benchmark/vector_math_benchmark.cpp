/** @file gridder_test_benchmark.cpp
 *  @brief Test Gridder module performance
 *
 *  @bug No known bugs.
 */
#include <benchmark/benchmark.h>
#include <cblas.h>
#include <common/vector_math.h>
#include <stp.h>

auto armadillo_accumulate_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(arma::accu(v));
    }
};

auto vector_accumulate_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_accumulate(v));
    }
};

auto vector_accumulate_parallel_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_accumulate_parallel(v));
    }
};

auto armadillo_mean_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(arma::mean(v));
    }
};

auto vector_mean_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_mean(v));
    }
};

auto vector_mean_parallel_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_mean_parallel(v));
    }
};

auto vector_mean_robust_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_mean_robust(v));
    }
};

auto vector_mean_robust_parallel_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_mean_parallel(v));
    }
};

auto armadillo_stddev_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(arma::stddev(v));
    }
};

auto vector_stddev_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_stddev(v));
    }
};

auto vector_stddev_parallel_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_stddev_parallel(v));
    }
};

auto vector_stddev_robust_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_stddev_robust(v));
    }
};

auto vector_stddev_robust_parallel_benchmark = [](benchmark::State& state, arma::vec v) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(stp::vector_stddev_parallel(v));
    }
};

auto matrix_arma_inplace_div_benchmark = [](benchmark::State& state, arma::cx_vec cv, double a) {
    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(cv /= a);
    }
};

auto matrix_tbb_inplace_div_benchmark = [](benchmark::State& state, arma::cx_vec cv, double a) {
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

auto matrix_cblas_inplace_div_benchmark = [](benchmark::State& state, arma::cx_vec cv, double a) {
    while (state.KeepRunning()) {
        uint n_elem = cv.n_elem;
        benchmark::DoNotOptimize(cv.memptr());
        cblas_zdscal(n_elem, (1 / a), cv.memptr(), 1); // for division we use (1 / a)
        benchmark::ClobberMemory();
    }
};

auto arma_shift_benchmark = [](benchmark::State& state) {
    int image_size = state.range(0);
    arma::arma_rng::set_seed(1);
    arma::cx_mat m = arma::randu<arma::cx_mat>(image_size, image_size);
    arma::cx_mat out;

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(out);
        out = arma::shift(m, state.range(1), state.range(2));
        benchmark::ClobberMemory();
    }
};

auto matrix_shift_benchmark = [](benchmark::State& state) {
    int image_size = state.range(0);
    arma::arma_rng::set_seed(1);
    arma::cx_mat m = arma::randu<arma::cx_mat>(image_size, image_size);
    arma::cx_mat out;

    while (state.KeepRunning()) {
        benchmark::DoNotOptimize(out);
        out = stp::matrix_shift(m, state.range(1), state.range(2));
        benchmark::ClobberMemory();
    }
};

arma::vec generate_vector_data(int image_size)
{
    arma::arma_rng::set_seed(1);
    return arma::randu<arma::vec>(image_size * image_size) + 10;
}

arma::cx_vec generate_cx_vector_data(int image_size)
{
    arma::arma_rng::set_seed(1);
    return arma::randu<arma::cx_vec>(image_size * image_size) + 10;
}

int main(int argc, char** argv)
{
    int image_size = 2048;
    arma::vec v = generate_vector_data(image_size);

    benchmark::RegisterBenchmark("armadillo_accumulate_benchmark", armadillo_accumulate_benchmark, v)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("vector_accumulate_benchmark", vector_accumulate_benchmark, v)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("vector_accumulate_parallel_benchmark", vector_accumulate_parallel_benchmark, v)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("armadillo_mean_benchmark", armadillo_mean_benchmark, v)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("vector_mean_benchmark", vector_mean_benchmark, v)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("vector_mean_parallel_benchmark", vector_mean_parallel_benchmark, v)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("vector_mean_robust_benchmark", vector_mean_robust_benchmark, v)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("vector_mean_robust_parallel_benchmark", vector_mean_robust_parallel_benchmark, v)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("armadillo_stddev_benchmark", armadillo_stddev_benchmark, v)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("vector_stddev_benchmark", vector_stddev_benchmark, v)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("vector_stddev_parallel_benchmark", vector_stddev_parallel_benchmark, v)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("vector_stddev_robust_benchmark", vector_stddev_robust_benchmark, v)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("vector_stddev_robust_parallel_benchmark", vector_stddev_robust_parallel_benchmark, v)->Unit(benchmark::kMicrosecond);

    double a = 0.03549;
    arma::cx_vec cv = generate_cx_vector_data(image_size);

    benchmark::RegisterBenchmark("matrix_arma_inplace_div_benchmark", matrix_arma_inplace_div_benchmark, cv, a)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("matrix_tbb_inplace_div_benchmark", matrix_tbb_inplace_div_benchmark, cv, a)->Unit(benchmark::kMicrosecond);
    benchmark::RegisterBenchmark("matrix_cblas_inplace_div_benchmark", matrix_cblas_inplace_div_benchmark, cv, a)->Unit(benchmark::kMicrosecond);

    benchmark::RegisterBenchmark("arma_shift_benchmark", arma_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ image_size, -image_size / 2, 0 })
        ->Args({ image_size, -image_size / 2, 1 })
        ->Args({ image_size, image_size / 2, 0 })
        ->Args({ image_size, image_size / 2, 1 });
    benchmark::RegisterBenchmark("matrix_shift_benchmark", matrix_shift_benchmark)
        ->Unit(benchmark::kMicrosecond)
        ->Args({ image_size, -image_size / 2, 0 })
        ->Args({ image_size, -image_size / 2, 1 })
        ->Args({ image_size, image_size / 2, 0 })
        ->Args({ image_size, image_size / 2, 1 });

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
