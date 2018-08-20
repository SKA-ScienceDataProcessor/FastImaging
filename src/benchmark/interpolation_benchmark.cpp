/** @file interpolation_benchmark.cpp
 *  @brief Test interpolation functions performance
 */

#include "../auxiliary/linear_interpolation.h"
#include "../stp/common/spline.h"
#include <benchmark/benchmark.h>
#include <stp.h>

void init_arrays(arma::Col<real_t>& x, arma::Col<real_t>& y, arma::Col<real_t>& xi, size_t length)
{
    size_t length_interp = length * 8;
    x = arma::regspace<arma::Col<real_t>>(0, length - 1);
    y = arma::randu(arma::size(x));
    xi = arma::sort(arma::randu(length_interp) * (length - 2));
}

static void arma_interpol1_benchmark(benchmark::State& state)
{
    size_t length = state.range(0);
    arma::Col<real_t> x, y, xi, yi;
    init_arrays(x, y, xi, length);
    yi.set_size(arma::size(xi));

    for (auto _ : state) {
        arma::interp1(x, y, xi, yi, "*linear");
    }
}

static void linear_interpolation_benchmark(benchmark::State& state)
{
    size_t length = state.range(0);
    arma::Col<real_t> x, y, xi, yi;
    init_arrays(x, y, xi, length);
    yi.set_size(arma::size(xi));

    for (auto _ : state) {
        for (size_t i = 0; i < xi.n_elem; ++i) {
            int lower_idx = int(xi[i]);
            yi[i] = linear_interpolation(x, y, xi[i], lower_idx);
        }
    }
}

static void spline_linear_benchmark(benchmark::State& state)
{
    size_t length = state.range(0);
    arma::Col<real_t> x, y, xi, yi;
    init_arrays(x, y, xi, length);
    yi.set_size(arma::size(xi));

    for (auto _ : state) {
        tk::spline<false> m_spline;
        m_spline.set_points(x, y, false);
        for (size_t i = 0; i < xi.n_elem; ++i) {
            int lower_idx = int(xi[i]);
            yi[i] = m_spline(xi[i], lower_idx, x[lower_idx], y[lower_idx]);
        }
    }
}

static void spline_cubic_benchmark(benchmark::State& state)
{
    size_t length = state.range(0);
    arma::Col<real_t> x, y, xi, yi;
    init_arrays(x, y, xi, length);
    yi.set_size(arma::size(xi));

    for (auto _ : state) {
        tk::spline<true> m_spline;
        m_spline.set_points(x, y, false);
        for (size_t i = 0; i < xi.n_elem; ++i) {
            int lower_idx = int(xi[i]);
            yi[i] = m_spline(xi[i], lower_idx, x[lower_idx], y[lower_idx]);
        }
    }
}

BENCHMARK(arma_interpol1_benchmark)->RangeMultiplier(2)->Range(1 << 10, 1 << 20)->Unit(benchmark::kMicrosecond);
BENCHMARK(linear_interpolation_benchmark)->RangeMultiplier(2)->Range(1 << 10, 1 << 20)->Unit(benchmark::kMicrosecond);
BENCHMARK(spline_linear_benchmark)->RangeMultiplier(2)->Range(1 << 10, 1 << 20)->Unit(benchmark::kMicrosecond);
BENCHMARK(spline_cubic_benchmark)->RangeMultiplier(2)->Range(1 << 10, 1 << 20)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
