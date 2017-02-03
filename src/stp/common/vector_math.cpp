#include "vector_math.h"
#include <math.h>
#include <tbb/tbb.h>
#include <thread>

namespace stp {

struct SumMean {
    double acc;
    arma::uword n_elem;

    SumMean()
        : acc(0.0)
        , n_elem(0)
    {
    }

    SumMean(double a, arma::uword ne)
        : acc(a)
        , n_elem(ne)
    {
    }
};

struct SumStdDev {
    double acc1;
    double acc2;
    arma::uword n_elem;

    SumStdDev()
        : acc1(0.0)
        , acc2(0.0)
        , n_elem(0)
    {
    }

    SumStdDev(double a1, double a2)
        : acc1(a1)
        , acc2(a2)
        , n_elem(0)
    {
    }

    SumStdDev(double a1, double a2, arma::uword num)
        : acc1(a1)
        , acc2(a2)
        , n_elem(num)
    {
    }
};

double vector_accumulate(arma::vec& v)
{
    double accu = 0.0;
    arma::uword v_size = v.n_elem;

    if (v_size > 0) {
        double acc1 = 0.0;
        double acc2 = 0.0;
        arma::uword i;
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
    double accu = 0.0;
    arma::uword v_size = v.n_elem;

    if (v_size > 0) {
        accu = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v_size), double(0.0), [&v](const tbb::blocked_range<size_t>& r, double sum) {
            for (size_t i = r.begin(); i != r.end(); i++) {
                sum += v.at(i);
            }
            return sum; },
            [](double x, double y) { return x + y; });
    }
    return accu;
}

double vector_mean(arma::vec& v)
{
    double mean = 0.0;
    double v_size = v.n_elem;

    if (v_size > 0) {
        mean = vector_accumulate(v) / v_size;
    }
    return mean;
}

double vector_mean_robust(arma::vec& v)
{
    double mean = 0.0;
    arma::uword v_size = v.n_elem;

    if (v_size > 0) {
        arma::uword n_elem = 0;
        double acc1 = 0.0;
        double acc2 = 0.0;
        arma::uword i;
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
        mean = (acc1 + acc2) / double(n_elem);
    }
    return mean;
}

double vector_mean_parallel(arma::vec& v)
{
    double mean = 0.0;
    double accu = 0.0;
    arma::uword v_size = v.n_elem;

    if (v_size > 0) {
        accu = vector_accumulate_parallel(v);
        mean = accu / v_size;
    }
    return mean;
}

double vector_mean_robust_parallel(arma::vec& v)
{
    double mean = 0.0;
    arma::uword v_size = v.n_elem;

    if (v_size > 0) {
        SumMean total = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v_size), SumMean(0, 0), [&v](const tbb::blocked_range<size_t>& r, SumMean sum) {
            double acc = sum.acc;
            arma::uword n_elem = sum.n_elem;
            for (size_t i = r.begin(); i != r.end(); i++) {
                if (arma::is_finite(v.at(i))) {
                    acc += v.at(i);
                    n_elem++;
                }
            }
            return SumMean(acc, n_elem); },
            [](SumMean x, SumMean y) { return SumMean(x.acc + y.acc, x.n_elem + y.n_elem); });
        mean = total.acc / double(total.n_elem);
    }
    return mean;
}

double vector_stddev(arma::vec& v, double mean)
{
    double std;
    arma::uword v_size = v.n_elem;

    if (v_size > 1) {
        if (!arma::is_finite(mean)) {
            mean = vector_mean(v);
        }
        double acc1 = 0;
        double acc2 = 0;
        for (arma::uword i = 0; i < v_size; i++) {
            const double tmp = v.at(i) - mean;
            acc1 += tmp * tmp;
            acc2 += tmp;
        }
        std = std::sqrt((acc1 - acc2 * acc2 / double(v_size)) / double(v_size));
    } else {
        std = 0.0;
    }

    return std;
}

double vector_stddev_robust(arma::vec& v, double mean)
{
    double std = 0.0;
    arma::uword v_size = v.n_elem;

    if (v_size > 1) {
        if (!arma::is_finite(mean)) {
            mean = vector_mean_robust(v);
        }
        arma::uword n_elem = 0;
        double acc1 = 0.0;
        double acc2 = 0.0;
        for (arma::uword i = 0; i < v_size; i++) {
            if (arma::is_finite(v.at(i))) {
                const double tmp = v.at(i) - mean;
                acc1 += tmp * tmp;
                acc2 += tmp;
                n_elem++;
            }
        }
        std = std::sqrt((acc1 - acc2 * acc2 / double(n_elem)) / double(n_elem));
    }

    return std;
}

double vector_stddev_parallel(arma::vec& v, double mean)
{
    double std = 0.0;
    arma::uword v_size = v.n_elem;

    if (v_size > 1) {
        if (!arma::is_finite(mean)) {
            mean = vector_mean_parallel(v);
        }
        SumStdDev total = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v_size), SumStdDev(0, 0), [&v, &mean](const tbb::blocked_range<size_t>& r, SumStdDev sum) {
            double acc1 = sum.acc1;
            double acc2 = sum.acc2;
            for (size_t i = r.begin(); i != r.end(); i++) {
                const double tmp = v.at(i) - mean;
                acc1 += tmp * tmp;
                acc2 += tmp;
            }
            return SumStdDev(acc1, acc2); },
            [](SumStdDev x, SumStdDev y) { return SumStdDev(x.acc1 + y.acc1, x.acc2 + y.acc2); });
        std = std::sqrt((total.acc1 - total.acc2 * total.acc2 / double(v_size)) / double(v_size));
    }

    return std;
}

double vector_stddev_robust_parallel(arma::vec& v, double mean)
{
    double std = 0.0;
    arma::uword v_size = v.n_elem;

    if (v_size > 1) {
        double l_mean = mean;
        if (!arma::is_finite(l_mean)) {
            l_mean = vector_mean_robust_parallel(v);
        }
        SumStdDev total = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, v_size), SumStdDev(0, 0, 0), [&v, &l_mean](const tbb::blocked_range<size_t>& r, SumStdDev sum) {
            double acc1 = sum.acc1;
            double acc2 = sum.acc2;
            arma::uword  n_elem = sum.n_elem;
            for (size_t i = r.begin(); i != r.end(); i++) {
                if (arma::is_finite(v.at(i))) {
                    const double tmp = v.at(i) - l_mean;
                    acc1 += tmp * tmp;
                    acc2 += tmp;
                    n_elem++;
                }
            }
            return SumStdDev(acc1, acc2, n_elem); },
            [](SumStdDev x, SumStdDev y) { return SumStdDev(x.acc1 + y.acc1, x.acc2 + y.acc2, x.n_elem + y.n_elem); });
        std = std::sqrt((total.acc1 - total.acc2 * total.acc2 / double(total.n_elem)) / double(total.n_elem));
    }

    return std;
}
}
