#include <cblas.h>
#include <common/matrix_math.h>
#include <fixtures.h>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

const double math_tolerance = 1.0e-5;

// Test matrix inplace division function
TEST(MatrixInplaceDivTest, MatrixInplaceDivCorrectness)
{
    long size = 512;
    arma::Mat<real_t> data = uncorrelated_gaussian_noise_background(size, size);
    double a = 0.3549;

    arma::Mat<real_t> div_arma = data;
    arma::Mat<real_t> div_tbb = data;
    arma::Mat<real_t> div_cblas = data;

    // Arma inplace division
    div_arma /= a;

    // TBB inplace division
    size_t n_elem = div_tbb.n_elem;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n_elem), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i < r.end(); i++) {
            div_tbb[i] /= a;
        }
    });

    // Cblas inplace division
    n_elem = div_cblas.n_elem;
#ifdef USE_FLOAT
    cblas_sscal((long)n_elem, (1.0f / a), div_cblas.memptr(), 1);
#else
    cblas_dscal(n_elem, (1.0 / a), div_cblas.memptr(), 1);
#endif

    EXPECT_TRUE(arma::approx_equal(div_arma, div_tbb, "absdiff", math_tolerance));
    EXPECT_TRUE(arma::approx_equal(div_arma, div_cblas, "absdiff", math_tolerance)); //We need to reduce tolerance value
}
