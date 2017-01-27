#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

// Sanity check everything works OK when input is 2d array (i.e. UV-coords)

const int oversampling_edge_case = 7;
const int oversampling_kernel_indices = 5;
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

TEST(GridderFractionalCoordIn2DCase, OversampledKernelIndices)
{
    arma::mat input = arma::join_rows(io_pairs.col(0), io_pairs.col(0));
    EXPECT_TRUE(arma::size(input) == arma::size(io_pairs));

    arma::imat output = calculate_oversampled_kernel_indices(input, oversampling_kernel_indices);
    EXPECT_TRUE(arma::size(input) == arma::size(output));
    EXPECT_EQ(arma::accu(io_pairs.col(1) != output.col(0)), 0);
    EXPECT_EQ(arma::accu(io_pairs.col(1) != output.col(1)), 0);
}
