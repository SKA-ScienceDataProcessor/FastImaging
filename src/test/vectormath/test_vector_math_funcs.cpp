/** @file conv_func_gaussian.cpp
 *  @brief Test Gaussian
 *
 *  TestCase to test the gaussian convolution function
 *  test with array input
 *
 *  @bug No known bugs.
 */

#include <cblas.h>
#include <common/vector_math.h>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

const double dtolerance = 1.0e-5;

arma::vec generate_vector_data()
{
    int image_size = 2048;
    arma::arma_rng::set_seed(1);
    return arma::randu<arma::vec>(image_size * image_size) + 10;
}

arma::cx_vec generate_cx_vector_data()
{
    int image_size = 2048;
    arma::arma_rng::set_seed(1);
    return arma::randu<arma::cx_vec>(image_size * image_size) + 10;
}

// Test the vector accumulate function
TEST(VectorAccumulateFunction, AccuValue)
{
    arma::vec v = generate_vector_data();
    double arma_accu = arma::accu(v);
    double vector_accu = stp::vector_accumulate(v);

    EXPECT_TRUE(abs(arma_accu - vector_accu) < dtolerance);
}

// Test the vector accumulate parallel function
TEST(VectorAccumulateParallelFunction, ParAccuValue)
{
    arma::vec v = generate_vector_data();
    double arma_accu = arma::accu(v);
    double vector_accu = stp::vector_accumulate_parallel(v);

    EXPECT_TRUE(abs(arma_accu - vector_accu) < dtolerance);
}

// Test the vector mean function
TEST(VectorMeanFunction, MeanValue)
{
    arma::vec v = generate_vector_data();
    double arma_mean = arma::mean(v);
    double vector_mean = stp::vector_mean(v);

    EXPECT_TRUE(abs(arma_mean - vector_mean) < dtolerance);
}

// Test the vector mean parallel function
TEST(VectorMeanParallelFunction, ParMeanValue)
{
    arma::vec v = generate_vector_data();
    double arma_mean = arma::mean(v);
    double vector_mean = stp::vector_mean_parallel(v);

    EXPECT_TRUE(abs(arma_mean - vector_mean) < dtolerance);
}

// Test the vector mean robust function
TEST(VectorMeanRobustFunction, RobustMeanValue)
{
    arma::vec v = generate_vector_data();
    double arma_mean = arma::mean(v);
    double vector_mean = stp::vector_mean_robust(v);

    EXPECT_TRUE(abs(arma_mean - vector_mean) < dtolerance);
}

// Test the vector mean robust parallel function
TEST(VectorMeanRobustParallelFunction, ParRobustMeanValue)
{
    arma::vec v = generate_vector_data();
    double arma_mean = arma::mean(v);
    double vector_mean = stp::vector_mean_robust_parallel(v);

    EXPECT_TRUE(abs(arma_mean - vector_mean) < dtolerance);
}

// Test the vector stddev function
TEST(VectorStddevFunction, StddevValue)
{
    arma::vec v = generate_vector_data();
    double arma_std = arma::stddev(v);
    double vector_std = stp::vector_stddev(v);

    EXPECT_TRUE(abs(arma_std - vector_std) < dtolerance);
}

// Test the vector stddev parallel function
TEST(VectorStddevParallelFunction, ParStddevValue)
{
    arma::vec v = generate_vector_data();
    double arma_std = arma::stddev(v);
    double vector_std = stp::vector_stddev_parallel(v);

    EXPECT_TRUE(abs(arma_std - vector_std) < dtolerance);
}

// Test the vector stddev robust function
TEST(VectorStddevRobustFunction, RobustStddevValue)
{
    arma::vec v = generate_vector_data();
    double arma_std = arma::stddev(v);
    double vector_std = stp::vector_stddev_robust(v);

    EXPECT_TRUE(abs(arma_std - vector_std) < dtolerance);
}

// Test the vector stddev robust parallel function
TEST(VectorStddevRobustParallelFunction, ParRobustStddevValue)
{
    arma::vec v = generate_vector_data();
    double arma_std = arma::stddev(v);
    double vector_std = stp::vector_stddev_robust_parallel(v);

    EXPECT_TRUE(abs(arma_std - vector_std) < dtolerance);
}

// Test matrix inplace division function
TEST(MatrixInplaceDivFunction, MatrixInplaceDiv)
{
    arma::cx_vec cv = generate_cx_vector_data();
    double a = 0.03549;

    arma::cx_vec v_div_arma = cv;
    arma::cx_vec v_div_tbb = cv;
    arma::cx_vec v_div_cblas = cv;

    // Arma inplace division
    v_div_arma /= a;

    // TBB inplace division
    uint n_elem = v_div_tbb.n_elem;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_elem), [&v_div_tbb, &a](const tbb::blocked_range<size_t>& r) {
        for (uint i = r.begin(); i != r.end(); i++) {
            v_div_tbb[i] /= a;
        }
    });

    // Cblas inplace division
    n_elem = v_div_cblas.n_elem;
    cblas_zdscal(n_elem, (1 / a), v_div_cblas.memptr(), 1);

    EXPECT_TRUE(arma::approx_equal(v_div_arma, v_div_tbb, "absdiff", tolerance));
    EXPECT_TRUE(arma::approx_equal(v_div_arma, v_div_cblas, "absdiff", 1.0e-13)); //We need to reduce tolerance value
}

// Test matrix shift function
TEST(MatrixShiftFunction, MatrixShift)
{

    int image_size = 2048;
    arma::arma_rng::set_seed(1);
    arma::cx_mat m = arma::randu<arma::cx_mat>(image_size, image_size);
    arma::cx_mat arma_out, matrix_out;

    arma_out = stp::matrix_shift(m, image_size / 2, 0);
    matrix_out = arma::shift(m, image_size / 2, 0);
    EXPECT_TRUE(arma::approx_equal(arma_out, matrix_out, "absdiff", tolerance));

    arma_out = stp::matrix_shift(m, image_size / 2, 1);
    matrix_out = arma::shift(m, image_size / 2, 1);
    EXPECT_TRUE(arma::approx_equal(arma_out, matrix_out, "absdiff", tolerance));
}
