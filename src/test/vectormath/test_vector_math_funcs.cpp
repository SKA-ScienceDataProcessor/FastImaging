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

#ifdef USE_FLOAT
const double dtolerance = 1.0e-4;
#else
const double dtolerance = 1.0e-5;
#endif

arma::Col<real_t> generate_vector_data()
{
    int image_size = 2048;
    arma::arma_rng::set_seed(1);
    return arma::randu<arma::Col<real_t> >(image_size * image_size) + 1;
}

arma::Col<cx_real_t> generate_cx_vector_data()
{
    int image_size = 2048;
    arma::arma_rng::set_seed(1);
    return arma::randu<arma::Col<cx_real_t> >(image_size * image_size) + 1;
}

// Test the vector accumulate function
TEST(VectorAccumulateFunction, AccuValue)
{
    arma::Col<real_t> v = generate_vector_data();
    double arma_accu = arma::accu(arma::conv_to<arma::Col<double> >::from(v));
    double vector_accu = stp::vector_accumulate(v);

    EXPECT_NEAR(arma_accu, vector_accu, dtolerance);
}

// Test the vector accumulate parallel function
TEST(VectorAccumulateParallelFunction, ParAccuValue)
{
    arma::Col<real_t> v = generate_vector_data();
    double arma_accu = arma::accu(arma::conv_to<arma::Col<double> >::from(v));
    double vector_accu = stp::vector_accumulate_parallel(v);

    EXPECT_NEAR(arma_accu, vector_accu, dtolerance);
}

// Test the vector mean function
TEST(VectorMeanFunction, MeanValue)
{
    arma::Col<real_t> v = generate_vector_data();
    double arma_mean = arma::mean(v);
    double vector_mean = stp::vector_mean(v);

    EXPECT_NEAR(arma_mean, vector_mean, dtolerance);
}

// Test the vector mean parallel function
TEST(VectorMeanParallelFunction, ParMeanValue)
{
    arma::Col<real_t> v = generate_vector_data();
    double arma_mean = arma::mean(v);
    double vector_mean = stp::vector_mean_parallel(v);

    EXPECT_NEAR(arma_mean, vector_mean, dtolerance);
}

// Test the vector mean robust function
TEST(VectorMeanRobustFunction, RobustMeanValue)
{
    arma::Col<real_t> v = generate_vector_data();
    double arma_mean = arma::mean(v);
    double vector_mean = stp::vector_mean_robust(v);

    EXPECT_NEAR(arma_mean, vector_mean, dtolerance);
}

// Test the vector mean robust parallel function
TEST(VectorMeanRobustParallelFunction, ParRobustMeanValue)
{
    arma::Col<real_t> v = generate_vector_data();
    double arma_mean = arma::mean(v);
    double vector_mean = stp::vector_mean_robust_parallel(v);

    EXPECT_NEAR(arma_mean, vector_mean, dtolerance);
}

// Test the vector stddev function
TEST(VectorStddevFunction, StddevValue)
{
    arma::Col<real_t> v = generate_vector_data();
    double arma_std = arma::stddev(v);
    double vector_std = stp::vector_stddev(v);

    EXPECT_NEAR(arma_std, vector_std, dtolerance);
}

// Test the vector stddev parallel function
TEST(VectorStddevParallelFunction, ParStddevValue)
{
    arma::Col<real_t> v = generate_vector_data();
    double arma_std = arma::stddev(v);
    double vector_std = stp::vector_stddev_parallel(v);

    EXPECT_NEAR(arma_std, vector_std, dtolerance);
}

// Test the vector stddev robust function
TEST(VectorStddevRobustFunction, RobustStddevValue)
{
    arma::Col<real_t> v = generate_vector_data();
    double arma_std = arma::stddev(v);
    double vector_std = stp::vector_stddev_robust(v);

    EXPECT_NEAR(arma_std, vector_std, dtolerance);
}

// Test the vector stddev robust parallel function
TEST(VectorStddevRobustParallelFunction, ParRobustStddevValue)
{
    arma::Col<real_t> v = generate_vector_data();
    double arma_std = arma::stddev(v);
    double vector_std = stp::vector_stddev_robust_parallel(v);

    EXPECT_NEAR(arma_std, vector_std, dtolerance);
}

// Test matrix inplace division function
TEST(MatrixInplaceDivFunction, MatrixInplaceDiv)
{
    arma::Col<cx_real_t> cv = generate_cx_vector_data();
    double a = 0.03549;

    arma::Col<cx_real_t> v_div_arma = cv;
    arma::Col<cx_real_t> v_div_tbb = cv;
    arma::Col<cx_real_t> v_div_cblas = cv;

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
#ifdef USE_FLOAT
    cblas_csscal(n_elem, (1.0 / a), reinterpret_cast<real_t*>(v_div_cblas.memptr()), 1);
#else
    cblas_zdscal(n_elem, (1.0 / a), reinterpret_cast<real_t*>(v_div_cblas.memptr()), 1);
#endif

    EXPECT_TRUE(arma::approx_equal(v_div_arma, v_div_tbb, "absdiff", dtolerance));
    EXPECT_TRUE(arma::approx_equal(v_div_arma, v_div_cblas, "absdiff", dtolerance)); //We need to reduce tolerance value
}

// Test matrix shift function
TEST(MatrixShiftFunction, MatrixShift)
{

    int image_size = 2048;
    arma::arma_rng::set_seed(1);
    arma::Mat<cx_real_t> m = arma::randu<arma::Mat<cx_real_t> >(image_size, image_size);
    arma::Mat<cx_real_t> arma_out, matrix_out;

    arma_out = stp::matrix_shift(m, image_size / 2, 0);
    matrix_out = arma::shift(m, image_size / 2, 0);
    EXPECT_TRUE(arma::approx_equal(arma_out, matrix_out, "absdiff", 0));

    arma_out = stp::matrix_shift(m, image_size / 2, 1);
    matrix_out = arma::shift(m, image_size / 2, 1);
    EXPECT_TRUE(arma::approx_equal(arma_out, matrix_out, "absdiff", 0));
}
