#include "../../libstp/gridder/gridder.h"
#include "gtest/gtest.h"

const double oversampling_1(7);
const double oversampling_2(5);
arma::mat subpix_offset = {0.5};
arma::mat io_pairs =
    {
        {    -0.5, -2 },
        { -0.4999, -2 },
        {    -0.4, -2 },
        {   -0.35, -2 },
        {    -0.3, -2 },
        { -0.2999, -1 },
        {    -0.2, -1 },
        {   -0.11, -1 },
        {    -0.1,  0 }, 
        {   -0.09,  0 },
        {   -0.01,  0 },
        {     0.0,  0 }
    };

TEST(GridderFractionalCoordToOversampledIndexMath, IndexValueGreater1) {    
    EXPECT_EQ(round(oversampling_1 * subpix_offset)[0], 4);
}

TEST(GridderFractionalCoordToOversampledIndexMath, IndexValueGreater2) {
    EXPECT_EQ(calculate_oversampled_kernel_indices(subpix_offset, oversampling_1)[0], 3);
}

TEST(GridderFractionalCoordToOversampledIndexMath, EasyCalculation) {    
    int col(0);
    arma::mat aux;
    aux.set_size(io_pairs.n_rows);
    
    io_pairs.each_row([&aux, &col](arma::mat &r) {
        aux[col] = r[0];
        col++;
    });
    
    arma::mat outputs = calculate_oversampled_kernel_indices(aux, oversampling_2);
     
    col = 0;
    io_pairs.each_row([&aux, &col](arma::mat &r) {
        aux[col] = r[1];
        col++;
    });      
    EXPECT_TRUE(arma::approx_equal(aux, outputs, "absdiff", tolerance));

}

TEST(GridderFractionalCoordToOversampledIndexMath, EasyCalculationSymetry) {
    io_pairs *= -1;
    
    int col(0);
    arma::mat aux;
    aux.set_size(io_pairs.n_rows);
    
    io_pairs.each_row([&aux, &col](arma::mat &r) {
        aux[col] = r[0];
        col++;
    });
    
    arma::mat outputs = calculate_oversampled_kernel_indices(aux, oversampling_2);
    
    col = 0;
    io_pairs.each_row([&aux, &col](arma::mat &r) {
        aux[col] = r[1];
        col++;
    });    
    
    EXPECT_TRUE(arma::approx_equal(aux, outputs, "absdiff", tolerance));
}


TEST(GridderFractionalCoordToOversampledIndexMath, CoOrdinatePairs) {
    arma::mat inputs = {{0.3, 0.3}};
    arma::mat outputs = {{2, 2}};
    EXPECT_TRUE(arma::approx_equal(calculate_oversampled_kernel_indices(inputs, oversampling_2), outputs, "absdiff", tolerance));
}
