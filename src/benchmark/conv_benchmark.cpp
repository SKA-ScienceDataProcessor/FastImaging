/** @file conv_benchmark.cpp
 *  @brief Test Convolution module performance
 */

#include <benchmark/benchmark.h>
#include <stp.h>

static void generate_1D_TopHat(benchmark::State& state)
{
    real_t offset = 0.0;
    int support = state.range(0);
    int oversampling = 1;

    for (auto _ : state) {
        stp::TopHat kernel_creator(3.0);
        int array_size = 2 * support * oversampling + 1;
        int centre_idx = support * oversampling;
        arma::Col<real_t> distance_vec((arma::linspace<arma::Col<real_t>>(0, array_size - 1, array_size) - centre_idx) / oversampling);
        benchmark::DoNotOptimize(kernel_creator(distance_vec - offset));
    }
}

static void generate_1D_Triangle(benchmark::State& state)
{
    real_t offset = 0.0;
    int support = state.range(0);
    int oversampling = 1;

    for (auto _ : state) {
        stp::Triangle kernel_creator(2.0);
        int array_size = 2 * support * oversampling + 1;
        int centre_idx = support * oversampling;
        arma::Col<real_t> distance_vec((arma::linspace<arma::Col<real_t>>(0, array_size - 1, array_size) - centre_idx) / oversampling);
        benchmark::DoNotOptimize(kernel_creator(distance_vec - offset));
    }
}

static void generate_1D_Sinc(benchmark::State& state)
{
    real_t offset = 0.0;
    int support = state.range(0);
    int oversampling = 1;

    for (auto _ : state) {
        stp::Sinc kernel_creator;
        int array_size = 2 * support * oversampling + 1;
        int centre_idx = support * oversampling;
        arma::Col<real_t> distance_vec((arma::linspace<arma::Col<real_t>>(0, array_size - 1, array_size) - centre_idx) / oversampling);
        benchmark::DoNotOptimize(kernel_creator(distance_vec - offset));
    }
}

static void generate_1D_Gaussian(benchmark::State& state)
{
    real_t offset = 0.0;
    int support = state.range(0);
    int oversampling = 1;

    for (auto _ : state) {
        stp::Gaussian kernel_creator;
        int array_size = 2 * support * oversampling + 1;
        int centre_idx = support * oversampling;
        arma::Col<real_t> distance_vec((arma::linspace<arma::Col<real_t>>(0, array_size - 1, array_size) - centre_idx) / oversampling);
        benchmark::DoNotOptimize(kernel_creator(distance_vec - offset));
        benchmark::ClobberMemory();
    }
}

static void generate_1D_GaussianSinc(benchmark::State& state)
{
    real_t offset = 0.0;
    int support = state.range(0);
    int oversampling = 1;

    for (auto _ : state) {
        stp::GaussianSinc kernel_creator(support);
        int array_size = 2 * support * oversampling + 1;
        int centre_idx = support * oversampling;
        arma::Col<real_t> distance_vec((arma::linspace<arma::Col<real_t>>(0, array_size - 1, array_size) - centre_idx) / oversampling);
        benchmark::DoNotOptimize(kernel_creator(distance_vec - offset));
        benchmark::ClobberMemory();
    }
}

static void generate_1D_PSWF(benchmark::State& state)
{
    real_t offset = 0.0;
    int support = state.range(0);
    int oversampling = 1;

    for (auto _ : state) {
        stp::PSWF kernel_creator(support);
        int array_size = 2 * support * oversampling + 1;
        int centre_idx = support * oversampling;
        arma::Col<real_t> distance_vec((arma::linspace<arma::Col<real_t>>(0, array_size - 1, array_size) - centre_idx) / oversampling);
        benchmark::DoNotOptimize(kernel_creator(distance_vec - offset));
        benchmark::ClobberMemory();
    }
}

BENCHMARK(generate_1D_TopHat)
    ->Args({ 3 })
    ->Args({ 5 })
    ->Args({ 7 });
BENCHMARK(generate_1D_Triangle)
    ->Args({ 3 })
    ->Args({ 5 })
    ->Args({ 7 });
BENCHMARK(generate_1D_Sinc)
    ->Args({ 3 })
    ->Args({ 5 })
    ->Args({ 7 });
BENCHMARK(generate_1D_Gaussian)
    ->Args({ 3 })
    ->Args({ 5 })
    ->Args({ 7 });
BENCHMARK(generate_1D_GaussianSinc)
    ->Args({ 3 })
    ->Args({ 5 })
    ->Args({ 7 });
BENCHMARK(generate_1D_PSWF)
    ->Args({ 3 })
    ->Args({ 5 })
    ->Args({ 7 });

BENCHMARK_MAIN();
