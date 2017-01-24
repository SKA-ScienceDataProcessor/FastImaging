#include "vector_math.h"
#include <math.h>
#include <tbb/tbb.h>
#include <thread>

namespace stp {

struct SumMean {
    double acc;
    uint n_elem;

    SumMean()
        : acc(0)
        , n_elem(0)
    {
    }

    SumMean(double a, uint ne)
        : acc(a)
        , n_elem(ne)
    {
    }
};

struct SumStdDev {
    double acc1;
    double acc2;
    uint n_elem;

    SumStdDev()
        : acc1(0)
        , acc2(0)
        , n_elem(0)
    {
    }

    SumStdDev(double a1, double a2)
        : acc1(a1)
        , acc2(a2)
        , n_elem(0)
    {
    }

    SumStdDev(double a1, double a2, uint num)
        : acc1(a1)
        , acc2(a2)
        , n_elem(num)
    {
    }
};

double vector_accumulate(arma::vec& v)
{
    double accu = 0;
    uint v_size = v.n_elem;

    if (v_size > 0) {
        double acc1 = 0;
        double acc2 = 0;
        uint i;
        for (i = 0; i < (v_size - 1); i += 2) {
            acc1 += v.at(i);
            acc2 += v.at(i + 1);
        }
        if (i < v_size) {
            acc1 += v.at(i);
        }
        accu = (acc1 + acc2);
    }
    return accu;
}

double vector_accumulate_parallel(arma::vec& v)
{
    double accu = 0;
    uint v_size = v.n_elem;

    if (v_size > 0) {
        accu = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v_size), double(0), [&v](const tbb::blocked_range<size_t>& r, double sum) {
            for (uint i = r.begin(); i != r.end(); i++) {
                sum += v.at(i);
            }
            return sum; },
            [](double x, double y) { return x + y; });
    }
    return accu;
}

double vector_mean(arma::vec& v)
{
    double mean = 0;
    uint v_size = v.n_elem;

    if (v_size > 0) {
        mean = vector_accumulate(v) / v_size;
    }
    return mean;
}

double vector_mean_robust(arma::vec& v)
{
    double mean = 0;
    uint v_size = v.n_elem;

    if (v_size > 0) {
        uint n_elem = 0;
        double acc1 = 0;
        double acc2 = 0;
        uint i;
        for (i = 0; i < (v_size - 1); i += 2) {
            if (arma::is_finite(v.at(i))) {
                acc1 += v.at(i);
                n_elem++;
            }
            if (arma::is_finite(v.at(i + 1))) {
                acc2 += v.at(i + 1);
                n_elem++;
            }
        }
        if (i < v_size) {
            if (arma::is_finite(v.at(i))) {
                acc1 += v.at(i);
                n_elem++;
            }
        }
        mean = (acc1 + acc2) / n_elem;
    }
    return mean;
}

double vector_mean_parallel(arma::vec& v)
{
    double mean = 0;
    double accu = 0;
    uint v_size = v.n_elem;

    if (v_size > 0) {
        accu = vector_accumulate_parallel(v);
        mean = accu / v_size;
    }
    return mean;
}

double vector_mean_robust_parallel(arma::vec& v)
{
    double mean = 0;
    uint v_size = v.n_elem;

    if (v_size > 0) {
        SumMean total = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v_size), SumMean(0, 0), [&v](const tbb::blocked_range<size_t>& r, SumMean sum) {
            double acc = sum.acc;
            uint n_elem = sum.n_elem;
            for (uint i = r.begin(); i != r.end(); i++) {
                if (arma::is_finite(v.at(i))) {
                    acc += v.at(i);
                    n_elem++;
                }
            }
            return SumMean(acc, n_elem); },
            [](SumMean x, SumMean y) { return SumMean(x.acc + y.acc, x.n_elem + y.n_elem); });
        mean = total.acc / total.n_elem;
    }
    return mean;
}

double vector_stddev(arma::vec& v, double mean)
{
    double std;
    uint v_size = v.n_elem;

    if (v_size > 1) {
        if (!arma::is_finite(mean)) {
            mean = vector_mean(v);
        }
        double acc1 = 0;
        double acc2 = 0;
        for (uint i = 0; i < v_size; i++) {
            const double tmp = v.at(i) - mean;
            acc1 += tmp * tmp;
            acc2 += tmp;
        }
        std = std::sqrt((acc1 - acc2 * acc2 / v_size) / v_size);
    } else {
        std = 0;
    }

    return std;
}

double vector_stddev_robust(arma::vec& v, double mean)
{
    double std;
    uint v_size = v.n_elem;

    if (v_size > 1) {
        if (!arma::is_finite(mean)) {
            mean = vector_mean_robust(v);
        }
        uint n_elem = 0;
        double acc1 = 0;
        double acc2 = 0;
        for (uint i = 0; i < v_size; i++) {
            if (arma::is_finite(v.at(i))) {
                const double tmp = v.at(i) - mean;
                acc1 += tmp * tmp;
                acc2 += tmp;
                n_elem++;
            }
        }
        std = std::sqrt((acc1 - acc2 * acc2 / n_elem) / n_elem);
    } else {
        std = 0;
    }

    return std;
}

double vector_stddev_parallel(arma::vec& v, double mean)
{
    double std;
    uint v_size = v.n_elem;

    if (v_size > 1) {
        if (!arma::is_finite(mean)) {
            mean = vector_mean_parallel(v);
        }
        SumStdDev total = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v_size), SumStdDev(0, 0), [&v, &mean](const tbb::blocked_range<size_t>& r, SumStdDev sum) {
            double acc1 = sum.acc1;
            double acc2 = sum.acc2;
            for (uint i = r.begin(); i != r.end(); i++) {
                const double tmp = v.at(i) - mean;
                acc1 += tmp * tmp;
                acc2 += tmp;
            }
            return SumStdDev(acc1, acc2); },
            [](SumStdDev x, SumStdDev y) { return SumStdDev(x.acc1 + y.acc1, x.acc2 + y.acc2); });
        std = std::sqrt((total.acc1 - total.acc2 * total.acc2 / v_size) / v_size);
    } else {
        std = 0;
    }

    return std;
}

double vector_stddev_robust_parallel(arma::vec& v, double mean)
{
    double std;
    uint v_size = v.n_elem;

    if (v_size > 1) {
        double l_mean = mean;
        if (!arma::is_finite(l_mean)) {
            l_mean = vector_mean_robust_parallel(v);
        }
        SumStdDev total = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v_size), SumStdDev(0, 0, 0), [&v, &l_mean](const tbb::blocked_range<size_t>& r, SumStdDev sum) {
            double acc1 = sum.acc1;
            double acc2 = sum.acc2;
            uint n_elem = sum.n_elem;
            for (uint i = r.begin(); i != r.end(); i++) {
                if (arma::is_finite(v.at(i))) {
                    const double tmp = v.at(i) - l_mean;
                    acc1 += tmp * tmp;
                    acc2 += tmp;
                    n_elem++;
                }
            }
            return SumStdDev(acc1, acc2, n_elem); },
            [](SumStdDev x, SumStdDev y) { return SumStdDev(x.acc1 + y.acc1, x.acc2 + y.acc2, x.n_elem + y.n_elem); });
        std = std::sqrt((total.acc1 - total.acc2 * total.acc2 / total.n_elem) / total.n_elem);
    } else {
        std = 0;
    }

    return std;
}

arma::cx_mat matrix_shift(const arma::cx_mat& in, const int length, const int dim)
{
    arma::cx_mat out(arma::size(in));

    // Use arma shift if the number of elements is not large enough for parallelization
    if (in.n_elem < 8192) {
        return arma::shift(in, length, dim);
    }

    const uint n_rows = in.n_rows;
    const uint n_cols = in.n_cols;
    const uint neg = length < 0 ? 1 : 0;
    const uint len = length < 0 ? (-length) : length;

    if (dim == 0) {
        if (neg == 0) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n_cols), [&out, &in, &len, &n_rows, &n_cols](const tbb::blocked_range<size_t>& r) {
                for (uint col = r.begin(); col != r.end(); ++col) {
                    arma::cx_double* out_ptr = out.colptr(col);
                    const arma::cx_double* in_ptr = in.colptr(col);

                    for (uint out_row = len, row = 0; row < (n_rows - len); ++row, ++out_row) {
                        out_ptr[out_row] = in_ptr[row];
                    }

                    for (uint out_row = 0, row = (n_rows - len); row < n_rows; ++row, ++out_row) {
                        out_ptr[out_row] = in_ptr[row];
                    }

                    //std::memcpy(out_ptr + len, in_ptr, (n_rows - len) * sizeof(arma::cx_double));
                    //std::memcpy(out_ptr, in_ptr + (n_rows - len), len * sizeof(arma::cx_double));
                }
            });
        } else if (neg == 1) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n_cols), [&out, &in, &len, &n_rows, &n_cols](const tbb::blocked_range<size_t>& r) {
                for (uint col = r.begin(); col != r.end(); ++col) {
                    arma::cx_double* out_ptr = out.colptr(col);
                    const arma::cx_double* in_ptr = in.colptr(col);

                    for (uint out_row = 0, row = len; row < n_rows; ++row, ++out_row) {
                        out_ptr[out_row] = in_ptr[row];
                    }

                    for (uint out_row = (n_rows - len), row = 0; row < len; ++row, ++out_row) {
                        out_ptr[out_row] = in_ptr[row];
                    }

                    //std::memcpy(out_ptr, in_ptr + len, (n_rows - len) * sizeof(arma::cx_double));
                    //std::memcpy(out_ptr + (n_rows - len), in_ptr, len * sizeof(arma::cx_double));
                }
            });
        }
    } else if (dim == 1) {
        if (neg == 0) {
            if (n_rows == 1) {
                arma::cx_double* out_ptr = out.memptr();
                const arma::cx_double* in_ptr = in.memptr();

                for (uint out_col = len, col = 0; col < (n_cols - len); ++col, ++out_col) {
                    out_ptr[out_col] = in_ptr[col];
                }

                for (uint out_col = 0, col = (n_cols - len); col < n_cols; ++col, ++out_col) {
                    out_ptr[out_col] = in_ptr[col];
                }
            } else {
                // Only parallelize if len >= std::thread::hardware_concurrency()*2
                if (len < std::thread::hardware_concurrency() * 2) {
                    for (uint out_col = len, col = 0; col < (n_cols - len); ++col, ++out_col) {
                        std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(arma::cx_double));
                    }

                    for (uint out_col = 0, col = (n_cols - len); col < n_cols; ++col, ++out_col) {
                        std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(arma::cx_double));
                    }
                } else {
                    tbb::parallel_for(tbb::blocked_range<size_t>(0, (n_cols - len)), [&out, &in, &len, &n_rows, &n_cols](const tbb::blocked_range<size_t>& r) {
                        for (uint out_col = r.begin() + len, col = r.begin(); col != r.end(); ++col, ++out_col) {
                            std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(arma::cx_double));
                        }
                    });
                    tbb::parallel_for(tbb::blocked_range<size_t>((n_cols - len), n_cols), [&out, &in, &len, &n_rows, &n_cols](const tbb::blocked_range<size_t>& r) {
                        for (uint out_col = r.begin() - (n_cols - len), col = r.begin(); col != r.end(); ++col, ++out_col) {
                            std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(arma::cx_double));
                        }
                    });
                }
            }
        } else if (neg == 1) {
            if (n_rows == 1) {
                arma::cx_double* out_ptr = out.memptr();
                const arma::cx_double* in_ptr = in.memptr();

                for (uint out_col = 0, col = len; col < n_cols; ++col, ++out_col) {
                    out_ptr[out_col] = in_ptr[col];
                }

                for (uint out_col = (n_cols - len), col = 0; col < len; ++col, ++out_col) {
                    out_ptr[out_col] = in_ptr[col];
                }
            } else {
                // Only parallelize if len >= std::thread::hardware_concurrency()*2
                if (len < std::thread::hardware_concurrency() * 2) {
                    for (uint out_col = 0, col = len; col < n_cols; ++col, ++out_col) {
                        std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(arma::cx_double));
                    }

                    for (uint out_col = (n_cols - len), col = 0; col < len; ++col, ++out_col) {
                        std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(arma::cx_double));
                    }
                } else {
                    tbb::parallel_for(tbb::blocked_range<size_t>(len, n_cols), [&out, &in, &len, &n_rows, &n_cols](const tbb::blocked_range<size_t>& r) {
                        for (uint out_col = r.begin() - len, col = r.begin(); col != r.end(); ++col, ++out_col) {
                            std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(arma::cx_double));
                        }
                    });
                    tbb::parallel_for(tbb::blocked_range<size_t>(0, len), [&out, &in, &len, &n_rows, &n_cols](const tbb::blocked_range<size_t>& r) {
                        for (uint out_col = r.begin() + (n_cols - len), col = r.begin(); col != r.end(); ++col, ++out_col) {
                            std::memcpy(out.colptr(out_col), in.colptr(col), n_rows * sizeof(arma::cx_double));
                        }
                    });
                }
            }
        }
    }
    return out;
}
}
