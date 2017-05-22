#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

#include "../types.h"
#include <armadillo>
#include <tbb/tbb.h>

// Minimum number elements required to use parallel implementation of shift.
// This is because smaller matrices do not benefit from parallel shift implementation.
#define MIN_ELEMS_FOR_PARSHIFT 8192

namespace stp {

/**
 * @brief Accumulate vector elements (single thread).
 *
 * Single threaded implementation of vector elements' accumulation.
 * All vector elements must be finite.
 *
 * @param[in] v (arma::vec): Input vector with elements to be accumulated.
 *
 * @return (double): Accumulation value.
 */
double vector_accumulate(arma::Col<real_t>& v);

/**
 * @brief Accumulate vector elements (parallel implementation).
 *
 * Parallel implementation of vector elements' accumulation.
 * All vector elements must be finite.
 *
 * @param[in] v (arma::vec): Input vector with elements to be accumulated.
 *
 * @return (double): Accumulation value.
 */
double vector_accumulate_parallel(arma::Col<real_t>& v);

/**
 * @brief Compute vector mean (single thread).
 *
 * Single threaded implementation of vector mean.
 * All vector elements must be finite.
 *
 * @param[in] v (arma::vec): Input vector.
 *
 * @return (double): Mean value of input vector.
 */
double vector_mean(arma::Col<real_t>& v);

/**
 * @brief Compute vector mean (parallel).
 *
 * Parallel implementation of vector mean.
 * All vector elements must be finite.
 *
 * @param[in] v (arma::vec): Input vector.
 *
 * @return (double): Mean value of input vector.
 */
double vector_mean_parallel(arma::Col<real_t>& v);

/**
 * @brief Compute mean value of finite vector elements (single thread).
 *
 * Single threaded implementation of vector mean.
 * Non-finite vector elements are not considered for mean calculation.
 *
 * @param[in] v (arma::vec): Input vector.
 *
 * @return (double): Mean value of input vector.
 */
double vector_mean_robust(arma::Col<real_t>& v);

/**
 * @brief Compute mean value of finite vector elements (parallel).
 *
 * Parallel implementation of vector mean.
 * Non-finite vector elements are not considered for mean calculation.
 *
 * @param[in] v (arma::vec): Input vector.
 *
 * @return (double): Mean value of input vector.
 */
double vector_mean_robust_parallel(arma::Col<real_t>& v);

/**
 * @brief Compute vector standard deviation (single thread).
 *
 * Single threaded implementation of vector standard deviation.
 * All vector elements must be finite.
 * If a pre-computed mean value is received, mean calculation within this function is bypassed.
 *
 * @param[in] v (arma::vec): Input vector.
 * @param[in] double (mean): Pre-computed mean value (optional).
 *
 * @return (double): Standard deviation of input vector.
 */
double vector_stddev(arma::Col<real_t>& v, double mean = arma::datum::nan);

/**
 * @brief Compute vector standard deviation (parallel).
 *
 * Parallel implementation of vector standard deviation.
 * All vector elements must be finite.
 * If a pre-computed mean value is received, mean calculation within this function is bypassed.
 *
 * @param[in] v (arma::vec): Input vector.
 * @param[in] double (mean): Pre-computed mean value (optional).
 *
 * @return (double): Standard deviation of input vector.
 */
double vector_stddev_parallel(arma::Col<real_t>& v, double mean = arma::datum::nan);

/**
 * @brief Compute standard deviation of finite vector elements (single thread).
 *
 * Single threaded implementation of vector standard deviation.
 * Non-finite vector elements are not considered for standard deviation calculation.
 * If a pre-computed mean value is received, mean calculation within this function is bypassed.
 *
 * @param[in] v (arma::vec): Input vector.
 * @param[in] double (mean): Pre-computed mean value (optional).
 *
 * @return (double): Standard deviation of input vector.
 */
double vector_stddev_robust(arma::Col<real_t>& v, double mean = arma::datum::nan);

/**
 * @brief Compute standard deviation of finite vector elements (parallel).
 *
 * Parallel implementation of vector standard deviation.
 * Non-finite vector elements are not considered for standard deviation calculation.
 * If a pre-computed mean value is received, mean calculation within this function is bypassed.
 *
 * @param[in] v (arma::vec): Input vector.
 * @param[in] double (mean): Pre-computed mean value (optional).
 *
 * @return (double): Standard deviation of input vector.
 */
double vector_stddev_robust_parallel(arma::Col<real_t>& v, double mean = arma::datum::nan);

/**
 * @brief Shift elements of input complex matrix in a circular manner.
 *
 * Based on armadillo shift implementation, but it uses TBB for parallel processing.
 * Template function allows to shift diferent matrix types, e.g. arma::mat, arma::cx_mat
 *
 * @param[in] m (arma::Mat<T>): Complex matrix to be shifted.
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
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n_cols), [&out, &in, &len, &n_rows, &n_cols](const tbb::blocked_range<size_t>& r) {
                for (size_t col = r.begin(); col != r.end(); ++col) {
                    T* out_ptr = out.colptr(col);
                    const T* in_ptr = in.colptr(col);

                    std::memcpy(out_ptr, in_ptr + (n_rows - len), len * sizeof(T));
                    std::memcpy(out_ptr + len, in_ptr, (n_rows - len) * sizeof(T));
                }
            });
        } else if (neg == 1) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n_cols), [&out, &in, &len, &n_rows, &n_cols](const tbb::blocked_range<size_t>& r) {
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
                tbb::parallel_for(tbb::blocked_range<size_t>(0, (n_cols - len)), [&out, &in, &len, &n_rows, &n_cols](const tbb::blocked_range<size_t>& r) {
                    for (size_t out_col = r.begin() + len, col = r.begin(); col != r.end(); ++col, ++out_col) {
                        std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(T));
                    }
                });
                tbb::parallel_for(tbb::blocked_range<size_t>((n_cols - len), n_cols), [&out, &in, &len, &n_rows, &n_cols](const tbb::blocked_range<size_t>& r) {
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
                tbb::parallel_for(tbb::blocked_range<size_t>(len, n_cols), [&out, &in, &len, &n_rows, &n_cols](const tbb::blocked_range<size_t>& r) {
                    for (size_t out_col = r.begin() - len, col = r.begin(); col != r.end(); ++col, ++out_col) {
                        std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(T));
                    }
                });
                tbb::parallel_for(tbb::blocked_range<size_t>(0, len), [&out, &in, &len, &n_rows, &n_cols](const tbb::blocked_range<size_t>& r) {
                    for (size_t out_col = r.begin() + (n_cols - len), col = r.begin(); col != r.end(); ++col, ++out_col) {
                        std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(T));
                    }
                });
            }
        }
    }
    return out;
}
}

#endif /* VECTOR_MATH_H */
