#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <libstp.h>

int support(3);
double half_base_width(2.5);
double triangle_value(1.0);
const double oversampling_cache(5);
const double oversampling_kernel(oversampling_disabled);
bool pad(false);
bool normalize(true);

void run()
{
    arma::mat steps = {
        { -0.4, 0.2, 0.0, 0.2, 0.4 }
    };
    arma::mat substeps = arma::linspace(-0.099999, 0.099999, 50);

    Triangle triangle(half_base_width, triangle_value);
    std::map<std::pair<int, int>, arma::mat> kernel_cache = populate_kernel_cache(support, oversampling_cache, pad, normalize, triangle);

    for (arma::uword i(0); i < steps.n_elem; ++i) {
        arma::mat x_offset = { steps[i], 0.0 };
        arma::mat aligned_exact_kernel = make_kernel_array(support, x_offset, oversampling_kernel, pad, normalize, triangle);
        arma::mat aligned_cache_idx = calculate_oversampled_kernel_indices(x_offset, oversampling_cache);
        arma::mat cached_kernel = kernel_cache[std::make_pair(aligned_cache_idx.at(0, 0), aligned_cache_idx.at(0, 1))];

        EXPECT_TRUE(arma::approx_equal(aligned_exact_kernel, cached_kernel, "absdiff", tolerance));

        for (arma::uword j(0); j < substeps.n_elem; ++j) {
            arma::mat offset = { x_offset[0] + substeps[j], 0.0 };
            if (substeps[j] < 0.0 || 0.0 < substeps[j]) {
                arma::mat unaligned_cache_idx = calculate_oversampled_kernel_indices(offset, oversampling_cache);

                EXPECT_TRUE(arma::approx_equal(unaligned_cache_idx, aligned_cache_idx, "absdiff", tolerance));
            }
        }
    }
}

TEST(GridderSteppedVsExactconvolution, equal)
{
    run();
}

TEST(GridderSteppedVsExactconvolution, GridderSteppedVsExactconvolution_benchmark)
{
    benchmark::RegisterBenchmark("SteppedVsExactConvolution", [](benchmark::State& state) { while(state.KeepRunning())run(); });
    benchmark::RunSpecifiedBenchmarks();
}
