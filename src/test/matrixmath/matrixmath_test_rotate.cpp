#include <common/matrix_math.h>
#include <fixtures.h>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

const double diff_tolerance = 1.0e-5;

TEST(MatrixRotateNoScale, EyeRotate90)
{
    int size = 11;
    arma::Mat<real_t> input = arma::eye<arma::Mat<real_t>>(size, size);
    arma::Mat<real_t> output = rotate_matrix(input, M_PI / 2, 0.0);

    EXPECT_NEAR(arma::accu(arma::abs(arma::flipud(input) - output)), 0.0, diff_tolerance);
}

TEST(MatrixRotateNoScale, EyeRotate180)
{
    int size = 11;
    arma::Mat<real_t> input = arma::eye<arma::Mat<real_t>>(size, size);
    arma::Mat<real_t> output = rotate_matrix(input, M_PI, 0.0);

    EXPECT_NEAR(arma::accu(arma::abs(input - output)), 0.0, diff_tolerance);
}

TEST(MatrixRotateNoScale, OnesRotate90)
{
    int size = 11;
    arma::Mat<real_t> input = arma::ones<arma::Mat<real_t>>(size, size);
    arma::Mat<real_t> output = rotate_matrix(input, M_PI / 2, 0.0);

    EXPECT_NEAR(arma::accu(arma::abs(input - output)), 0.0, diff_tolerance);
}

TEST(MatrixRotateNoScale, OnesRotate45)
{
    int size = 5;
    arma::Mat<real_t> input = arma::ones<arma::Mat<real_t>>(size, size);
    arma::Mat<real_t> output = rotate_matrix(input, M_PI / 4, 0.0);
    arma::Mat<real_t> res = {
        { 0.0, 0.0, 1.0, 0.0, 0.0 },
        { 0.0, 1.0, 1.0, 1.0, 0.0 },
        { 1.0, 1.0, 1.0, 1.0, 1.0 },
        { 0.0, 1.0, 1.0, 1.0, 0.0 },
        { 0.0, 0.0, 1.0, 0.0, 0.0 }
    };

    EXPECT_NEAR(arma::accu(arma::abs(output - res)), 0.0, diff_tolerance);
}

TEST(MatrixRotateNoScale, SlineRotate180)
{
    int size = 5;
    arma::Mat<real_t> input = arma::zeros<arma::Mat<real_t>>(size, size);
    input(2, 2) = 1.0;
    input(2, 3) = 1.0;
    arma::Mat<real_t> output = rotate_matrix(input, M_PI, 0.0);

    arma::Mat<real_t> res = arma::zeros<arma::Mat<real_t>>(size, size);
    res(2, 2) = 1.0;
    res(2, 1) = 1.0;

    EXPECT_NEAR(arma::accu(arma::abs(output - res)), 0.0, diff_tolerance);
}

TEST(MatrixRotateNoScale, SemiOnesRotate180)
{
    int size = 4;
    arma::Mat<real_t> input = arma::zeros<arma::Mat<real_t>>(size, size);
    input.rows(2, 3) = arma::ones<arma::Mat<real_t>>(2, 4);
    arma::Mat<real_t> output = rotate_matrix(input, M_PI, 0.0);

    EXPECT_NEAR(arma::accu(arma::abs(output - arma::flipud(input))), 0.0, diff_tolerance);
}

TEST(MatrixRotateNoScale, SemiOnesRotate90)
{
    int size = 4;
    arma::Mat<real_t> input = arma::zeros<arma::Mat<real_t>>(size, size);
    input.rows(2, 3) = arma::ones<arma::Mat<real_t>>(2, 4);
    arma::Mat<real_t> output = rotate_matrix(input, M_PI / 2, 0.0);

    EXPECT_NEAR(arma::accu(arma::abs(output - arma::flipud(input.st()))), 0.0, diff_tolerance);
}

TEST(MatrixRotateUpScale, OnesRotate0Size4To8)
{
    int size = 4;
    arma::Mat<real_t> input = arma::ones<arma::Mat<real_t>>(size, size);
    arma::Mat<real_t> output = rotate_matrix(input, 0.0, 0.0, 8);

    EXPECT_NEAR(arma::accu(arma::abs(output)), 36.0, diff_tolerance);
}

TEST(MatrixRotateUpScale, OnesRotate180Size4To5)
{
    int size = 4;
    arma::Mat<real_t> input = arma::ones<arma::Mat<real_t>>(size, size);
    arma::Mat<real_t> output = rotate_matrix(input, 0.0, 0.0, 5);

    EXPECT_NEAR(arma::accu(arma::abs(output)), 9.0, diff_tolerance);
}

TEST(MatrixRotateDownScale, OnesRotate180Size8To4)
{
    int size = 8;
    arma::Mat<real_t> input = arma::ones<arma::Mat<real_t>>(size, size);
    arma::Mat<real_t> output = rotate_matrix(input, M_PI, 0.0, 4);

    EXPECT_NEAR(arma::accu(arma::abs(output)), 16.0, diff_tolerance);
}

TEST(MatrixRotateDownScale, SemiOnesRotate180Size4To2)
{
    int size = 4;
    arma::Mat<real_t> input = arma::zeros<arma::Mat<real_t>>(size, size);
    input.rows(2, 3) = arma::ones<arma::Mat<real_t>>(2, 4);
    arma::Mat<real_t> output = rotate_matrix(input, M_PI, 0.0, 2);
    arma::Mat<real_t> res = { { 1.0, 1.0 }, { 0.0, 0.0 } };
    EXPECT_NEAR(arma::accu(arma::abs(output - res)), 0.0, diff_tolerance);
}
