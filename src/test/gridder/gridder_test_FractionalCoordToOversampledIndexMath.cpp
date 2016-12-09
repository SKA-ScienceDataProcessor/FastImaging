#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <libstp.h>

const double oversampling_edge_case(7);
const double oversampling_kernel_indices(5);
arma::mat subpix_offset = { 0.5 };
arma::mat io_pairs = {
    { -0.5, -2 },
    { -0.4999, -2 },
    { -0.4, -2 },
    { -0.35, -2 },
    { -0.3, -2 },
    { -0.2999, -1 },
    { -0.2, -1 },
    { -0.11, -1 },
    { -0.1, 0 },
    { -0.09, 0 },
    { -0.01, 0 },
    { 0.0, 0 }
};

TEST(GridderFractionalCoordToOversampledIndexMath, IndexValueGreaterHalfOversampling)
{
    EXPECT_EQ(round(oversampling_edge_case * subpix_offset)[0], 4);
}

TEST(GridderFractionalCoordToOversampledIndexMath, OversampledKernelIndices)
{
    EXPECT_EQ(calculate_oversampled_kernel_indices(subpix_offset, oversampling_edge_case)[0], 3);
}

TEST(GridderFractionalCoordToOversampledIndexMath, EasyCalculation)
{
    int col(0);
    arma::mat aux;
    aux.set_size(io_pairs.n_rows);

    io_pairs.each_row([&aux, &col](arma::mat& r) {
        aux[col] = r[0];
        col++;
    });

    arma::mat outputs = calculate_oversampled_kernel_indices(aux, oversampling_kernel_indices);

    col = 0;
    io_pairs.each_row([&aux, &col](arma::mat& r) {
        aux[col] = r[1];
        col++;
    });
    EXPECT_TRUE(arma::approx_equal(aux, outputs, "absdiff", tolerance));
}

TEST(GridderFractionalCoordToOversampledIndexMath, EasyCalculationSymetry)
{
    io_pairs *= -1;

    int col(0);
    arma::mat aux;
    aux.set_size(io_pairs.n_rows);

    io_pairs.each_row([&aux, &col](arma::mat& r) {
        aux[col] = r[0];
        col++;
    });

    arma::mat outputs = calculate_oversampled_kernel_indices(aux, oversampling_kernel_indices);

    col = 0;
    io_pairs.each_row([&aux, &col](arma::mat& r) {
        aux[col] = r[1];
        col++;
    });

    EXPECT_TRUE(arma::approx_equal(aux, outputs, "absdiff", tolerance));
}

TEST(GridderFractionalCoordToOversampledIndexMath, CoOrdinatePairs)
{
    arma::mat inputs = { { 0.3, 0.3 } };
    arma::mat outputs = { { 2, 2 } };
    EXPECT_TRUE(arma::approx_equal(calculate_oversampled_kernel_indices(inputs, oversampling_kernel_indices), outputs, "absdiff", tolerance));
}

// Benchmark functions

void BM_OversampledKernelIndices(benchmark::State& state)
{
    while (state.KeepRunning())
        calculate_oversampled_kernel_indices(subpix_offset, oversampling_edge_case);
}

TEST(GridderFractionalCoordToOversampledIndexMath, FractionalCoordToOversampledIndexMath_benchmark)
{
    BENCHMARK(BM_OversampledKernelIndices);
    benchmark::RunSpecifiedBenchmarks();
}
