/**
* @file matrix_math.h
* @brief Classes and function prototypes of matrix math.
*/

#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include "../types.h"
#include <armadillo>
#include <tbb/tbb.h>

// Minimum number elements required to use parallel implementation of shift.
// This is because smaller matrices do not benefit from parallel shift implementation.
#define MIN_ELEMS_FOR_PARSHIFT 8192

namespace stp {

/**
 * @brief Struct that stores computed statistics: mean, sigma, median.
 */
struct DataStats {

    /**
     * @brief DataStats default constructor.
     *
     * All statistic values set to zero.
     */
    DataStats()
        : mean(0.0)
        , sigma(0.0)
        , median(0.0)
        , mean_valid(false)
        , sigma_valid(false)
        , median_valid(false)
    {
    }

    /**
     * @brief DataStats constructor that sets given mean value. Sigma and median set to zero.
     *
     * @param[in] u (real_t): Mean value
     */
    DataStats(real_t u)
        : mean(u)
        , sigma(0.0)
        , median(0.0)
        , mean_valid(true)
        , sigma_valid(false)
        , median_valid(false)
    {
    }

    /**
     * @brief DataStats constructor that sets given mean and sigma values. Median set to zero.
     *
     * @param[in] u (real_t): Mean value
     * @param[in] s (real_t): Sigma value
     */
    DataStats(real_t u, real_t s)
        : mean(u)
        , sigma(s)
        , median(0.0)
        , mean_valid(true)
        , sigma_valid(true)
        , median_valid(false)
    {
    }

    /**
     * @brief DataStats constructor that sets given mean, sigma and median values.
     *
     * @param[in] u (real_t): Mean value
     * @param[in] s (real_t): Sigma value
     * @param[in] m (real_t): Median value
     */
    DataStats(real_t u, real_t s, real_t m)
        : mean(u)
        , sigma(s)
        , median(m)
        , mean_valid(true)
        , sigma_valid(true)
        , median_valid(true)
    {
    }

    // Mean value
    real_t mean;

    // Sigma value
    real_t sigma;

    // Median value
    real_t median;

    // Validity bits
    bool mean_valid;
    bool sigma_valid;
    bool median_valid;
};

/**
 * @brief Struct that stores a pair of doubles.
 */
struct DoublePair {
    double d1;
    double d2;

    /**
     * @brief DoublePair default constructor.
     *
     * Values set to zero.
     */
    DoublePair()
        : d1(0.0)
        , d2(0.0)
    {
    }

    /**
     * @brief DoublePair constructor that sets two double numbers.
     *
     * @param[in] a1 (double): First double value
     * @param[in] a2 (double): Second double value
     */
    DoublePair(double a1, double a2)
        : d1(a1)
        , d2(a2)
    {
    }
};

/**
 * @brief Compute exact median using the nth_element function.
 *
 * @param[in] data (arma::Mat): Input matrix.
 *
 * @return (double): Exact median.
 */
double mat_median_exact(const arma::Mat<real_t>& data);

/**
 * @brief Compute exact median using a method that combines sucessive binning and nth_element function.
 *
 * Provides parallel implementation of modified binmedian - a fast method to find exact median.
 * The method is based on binmedian algorithm (by Ryan Tibshirani) available at:
 * http://www.stat.cmu.edu/~ryantibs/median/
 *
 * @param[in] data (arma::Mat): Input matrix.
 *
 * @return (DataStats): Exact median, mean and sigma.
 */
DataStats mat_binmedian(const arma::Mat<real_t>& data);

/**
 * @brief Compute approximation of the median using the binapprox method.
 *
 * Provides parallel implementation of the binapprox method to compute an approximation of the median.
 * The median error is guaranteed to be inferior to sigma/1000.
 * An algorithm description and its paper (by Ryan Tibshirani) are available at:
 * http://www.stat.cmu.edu/~ryantibs/median/
 *
 * @param[in] data (arma::Mat): Input matrix.
 *
 * @return (DataStats): Approximation of the median value. Also returns mean and sigma.
 */
DataStats mat_median_binapprox(const arma::Mat<real_t>& data);

/**
 * @brief Compute matrix mean and standard deviation at once.
 *
 * Provides parallel implementation of matrix mean and standard deviation.
 * This function uses a single loop over data to compute both quantities at once and thus get better performance.
 * It does not use a numerically stable implementation.
 *
 * @param[in] data (arma::Mat): Input matrix.
 *
 * @return (DataStats): Mean and standard deviation values.
 */
DataStats mat_mean_and_stddev(const arma::Mat<real_t>& data);

/**
 * @brief Accumulate matrix elements (single thread implementation).
 *
 * Provides single threaded implementation of matrix elements' accumulation.
 *
 * @param[in] data (arma::Mat): Input matrix with elements to be accumulated.
 *
 * @return (double): Accumulation value.
 */
double mat_accumulate(arma::Mat<real_t>& data);

/**
 * @brief Accumulate matrix elements (parallel implementation).
 *
 * Provides parallel implementation of matrix elements' accumulation.
 *
 * @param[in] data (arma::Mat): Input matrix with elements to be accumulated.
 *
 * @return (double): Accumulation value.
 */
double mat_accumulate_parallel(arma::Mat<real_t>& data);

/**
 * @brief Compute matrix mean (single thread implementation).
 *
 * Provides single threaded implementation of matrix mean.
 *
 * @param[in] data (arma::Mat): Input matrix.
 *
 * @return (double): Mean value of input matrix.
 */
double mat_mean(arma::Mat<real_t>& data);

/**
 * @brief Compute matrix mean (parallel).
 *
 * Provides parallel implementation of matrix mean.
 *
 * @param[in] data (arma::Mat): Input matrix.
 *
 * @return (double): Mean value of input matrix.
 */
double mat_mean_parallel(arma::Mat<real_t>& data);

/**
 * @brief Compute matrix standard deviation (parallel).
 *
 * Provides parallel implementation of matrix standard deviation.
 * If a pre-computed mean value is received, mean calculation is bypassed.
 *
 * @param[in] data (arma::Mat): Input matrix.
 * @param[in] bool (compute_mean): Indicate if the mean value is an input or if it must be computed (default = true).
 * @param[in] double (mean): Pre-computed mean value (required only when compute_mean is false).
 *
 * @return (double): Standard deviation of input matrix.
 */
double mat_stddev_parallel(arma::Mat<real_t>& data, bool compute_mean = true, double mean = 0.0);

/**
 * @brief Shift elements of the input matrix in a circular manner.
 *
 * Provides matrix shift operation based on armadillo implementation, but it uses TBB for parallel processing.
 * Template function allows to shift diferent matrix types, e.g. arma::mat, arma::cx_mat
 *
 * @param[in] in (arma::Mat<T>): Matrix to be shifted.
 * @param[in] lenght (int): Number of positions to shifted (can be positive or negative).
 * @param[in] dim(int): If dim=0, shift each column by "lenght". If dim=1, shift each row by "lenght". Default is 0.
 *
 * @return (arma::Mat<T> ): Shifted version of input matrix.
 */
template <typename T>
arma::Mat<T> matrix_shift(const arma::Mat<T>& in, const int length, const int dim = 0)
{
    arma::Mat<T> out(arma::size(in));

    // Use arma shift if the number of elements is not large enough for parallelization
    if (in.n_elem < MIN_ELEMS_FOR_PARSHIFT) {
        return arma::shift(in, length, dim);
    }

    const arma::uword n_rows = in.n_rows;
    const arma::uword n_cols = in.n_cols;
    const arma::uword neg = length < 0 ? 1 : 0;
    const arma::uword len = length < 0 ? (-length) : length;

    if (dim == 0) {
        if (neg == 0) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n_cols), [&](const tbb::blocked_range<size_t>& r) {
                for (size_t col = r.begin(); col != r.end(); ++col) {
                    T* out_ptr = out.colptr(col);
                    const T* in_ptr = in.colptr(col);

                    std::memcpy(out_ptr, in_ptr + (n_rows - len), len * sizeof(T));
                    std::memcpy(out_ptr + len, in_ptr, (n_rows - len) * sizeof(T));
                }
            });
        } else if (neg == 1) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n_cols), [&](const tbb::blocked_range<size_t>& r) {
                for (size_t col = r.begin(); col != r.end(); ++col) {
                    T* out_ptr = out.colptr(col);
                    const T* in_ptr = in.colptr(col);

                    std::memcpy(out_ptr, in_ptr + len, (n_rows - len) * sizeof(T));
                    std::memcpy(out_ptr + (n_rows - len), in_ptr, len * sizeof(T));
                }
            });
        }
    } else if (dim == 1) {
        if (neg == 0) {
            if (n_rows == 1) {
                T* out_ptr = out.memptr();
                const T* in_ptr = in.memptr();

                std::memcpy(out_ptr, in_ptr + (n_cols - len), len * sizeof(T));
                std::memcpy(out_ptr + len, in_ptr, (n_cols - len) * sizeof(T));

            } else {
                tbb::parallel_for(tbb::blocked_range<size_t>(0, (n_cols - len)), [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t out_col = r.begin() + len, col = r.begin(); col != r.end(); ++col, ++out_col) {
                        std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(T));
                    }
                });
                tbb::parallel_for(tbb::blocked_range<size_t>((n_cols - len), n_cols), [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t out_col = r.begin() - (n_cols - len), col = r.begin(); col != r.end(); ++col, ++out_col) {
                        std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(T));
                    }
                });
            }
        } else if (neg == 1) {
            if (n_rows == 1) {
                T* out_ptr = out.memptr();
                const T* in_ptr = in.memptr();

                std::memcpy(out_ptr, in_ptr + len, (n_cols - len) * sizeof(T));
                std::memcpy(out_ptr + (n_cols - len), in_ptr, len * sizeof(T));

            } else {
                tbb::parallel_for(tbb::blocked_range<size_t>(len, n_cols), [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t out_col = r.begin() - len, col = r.begin(); col != r.end(); ++col, ++out_col) {
                        std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(T));
                    }
                });
                tbb::parallel_for(tbb::blocked_range<size_t>(0, len), [&](const tbb::blocked_range<size_t>& r) {
                    for (size_t out_col = r.begin() + (n_cols - len), col = r.begin(); col != r.end(); ++col, ++out_col) {
                        std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(T));
                    }
                });
            }
        }
    }
    return out;
}

arma::Mat<real_t> rotate_matrix(const arma::Mat<real_t>& in_m, double angle, double cval, int out_size = 0);
arma::Mat<cx_real_t> rotate_matrix(const arma::Mat<cx_real_t>& in_m, double angle, cx_real_t cval, int out_size = 0);
}

#endif /* MATRIX_MATH_H */
