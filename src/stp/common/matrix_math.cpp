#include "matrix_math.h"
#include <cassert>
#include <cblas.h>
#include <math.h>
#include <thread>

namespace stp {

DataStats mat_median_binapprox(const arma::Mat<real_t>& data)
{
    double median = arma::datum::nan;
    size_t num_elems = data.n_elem;
    auto m_stats = mat_mean_and_stddev(data);
    double mean = m_stats.mean;
    double sigma = m_stats.sigma;
    const uint N = 1000;

    // Bin data across the interval [mean-sigma, mean+sigma]
    tbb::combinable<size_t> bottomcount(0);
    tbb::combinable<arma::uvec> bincounts(arma::uvec(N + 1).zeros());
    const real_t scalefactor = N / (2 * sigma);
    const real_t leftend = mean - sigma;
    const real_t rightend = mean + sigma;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elems, 4), [&](const tbb::blocked_range<size_t>& r) {
        size_t& l_bottomcount = bottomcount.local();
        arma::uvec& l_bincounts = bincounts.local();
        size_t i = r.begin();
        size_t rend = r.end();
        // Process 2 elements per loop iteration, only when range size is larger than 4
        if ((rend - i) > 4) {
            rend -= 2;
            for (; i < rend; i += 2) {
                const real_t& val = data[i];
                const real_t& val2 = data[i + 1];
                if (val < leftend) {
                    l_bottomcount++;
                } else {
                    if (val < rightend) {
                        const int bin = (((val - leftend) * scalefactor));
                        l_bincounts[bin]++;
                    }
                }
                if (val2 < leftend) {
                    l_bottomcount++;
                } else {
                    if (val2 < rightend) {
                        const int bin = (((val2 - leftend) * scalefactor));
                        l_bincounts[bin]++;
                    }
                }
            }
            rend += 2;
        }
        for (; i < rend; i++) {
            const real_t& val = data[i];
            if (val < leftend) {
                l_bottomcount++;
            } else {
                if (val < rightend) {
                    const int bin = (((val - leftend) * scalefactor));
                    l_bincounts[bin]++;
                }
            }
        }
    });

    size_t bottomcount_all = bottomcount.combine([](const size_t& x, const size_t& y) { return x + y; });
    arma::uvec bincounts_all = bincounts.combine([](const arma::uvec& x, const arma::uvec& y) { return x + y; });

    // If size is odd
    if (num_elems & 1) {
        // Find the bin that contains the median
        size_t k = (num_elems + 1) / 2;
        size_t count = bottomcount_all;

        for (uint i = 0; i < (N + 1); i++) {
            count += bincounts_all.at(i);

            if (count >= k) {
                median = double(i + 0.5) / scalefactor + leftend;
                break;
            }
        }
    }
    // If size is even
    else {
        // Find the bins that contains the medians
        size_t k = num_elems / 2;
        size_t count = bottomcount_all;

        for (uint i = 0; i < (N + 1); i++) {
            count += bincounts_all.at(i);

            if (count >= k) {
                uint j = i;
                while (count == k) {
                    j++;
                    count += bincounts_all.at(j);
                }
                median = double(i + j + 1) / (2 * scalefactor) + leftend;
                break;
            }
        }
    }
    return DataStats(mean, sigma, median);
}

DataStats mat_mean_and_stddev(const arma::Mat<real_t>& data)
{
    double sigma = 0.0;
    double mean = 0.0;
    size_t num_elems = data.n_elem;

    if (num_elems < 1) {
        return DataStats(arma::datum::nan, arma::datum::nan);
    }

    DoublePair total = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, num_elems, 8), DoublePair(0.0, 0.0),
        [&](const tbb::blocked_range<size_t>& r, DoublePair sum) {
            size_t i = r.begin();
            size_t rend = r.end();
            // Process 4 elements per loop iteration, only when range size is larger than 8
            if ((rend - i) > 8) {
                rend -= 4;
                for (; i < rend; i += 4) {
                    const real_t& tmp = data[i];
                    const real_t& tmp2 = data[i + 1];
                    const real_t& tmp3 = data[i + 2];
                    const real_t& tmp4 = data[i + 3];
                    sum.d1 += tmp + tmp2 + tmp3 + tmp4;
                    sum.d2 += tmp * tmp + tmp2 * tmp2 + tmp3 * tmp3 + tmp4 * tmp4;
                }
                rend += 4;
            }
            for (; i < rend; i++) {
                const real_t& tmp = data[i];
                sum.d1 += tmp;
                sum.d2 += tmp * tmp;
            }

            return std::move(sum);
        },
        [](const DoublePair& x, const DoublePair& y) { return DoublePair(x.d1 + y.d1, x.d2 + y.d2); });
    mean = total.d1 / double(num_elems);
    sigma = std::sqrt(total.d2 / double(num_elems) - (mean * mean));

    return DataStats(mean, sigma);
}

double mat_accumulate(arma::Mat<real_t>& data)
{
    double accu = 0.0;
    arma::uword num_elems = data.n_elem;

    if (num_elems > 0) {
        double acc1 = 0.0;
        double acc2 = 0.0;
        arma::uword i;
        for (i = 0; i < (num_elems - 1); i += 2) {
            acc1 += data.at(i);
            acc2 += data.at(i + 1);
        }
        if (i < num_elems) {
            acc1 += data.at(i);
        }
        accu = (acc1 + acc2);
    }
    return accu;
}

double mat_accumulate_parallel(arma::Mat<real_t>& data)
{
    double accu = 0.0;
    arma::uword num_elems = data.n_elem;

    if (num_elems > 0) {
        accu = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, num_elems), double(0.0), [&](const tbb::blocked_range<size_t>& r, double sum) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                sum += data.at(i);
            }
            return sum; },
            [](double x, double y) { return x + y; });
    }
    return accu;
}

double mat_mean(arma::Mat<real_t>& data)
{
    double mean = 0.0;
    double num_elems = data.n_elem;

    if (num_elems > 0) {
        mean = mat_accumulate(data) / num_elems;
    }
    return mean;
}

double mat_mean_parallel(arma::Mat<real_t>& data)
{
    double mean = 0.0;
    double accu = 0.0;
    arma::uword num_elems = data.n_elem;

    if (num_elems > 0) {
        accu = mat_accumulate_parallel(data);
        mean = accu / num_elems;
    }
    return mean;
}

double mat_stddev_parallel(arma::Mat<real_t>& data, double mean)
{
    double std = 0.0;
    arma::uword num_elems = data.n_elem;

    if (num_elems > 1) {
        if (!arma::is_finite(mean)) {
            mean = mat_mean_parallel(data);
        }
        SumStdDev total = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, num_elems), SumStdDev(0.0, 0.0), [&](const tbb::blocked_range<size_t>& r, SumStdDev sum) {
            double& acc1 = sum.acc1;
            double& acc2 = sum.acc2;
            for (size_t i = r.begin(); i != r.end(); i++) {
                const double tmp = data.at(i) - mean;
                acc1 += tmp * tmp;
                acc2 += tmp;
            }
            return SumStdDev(acc1, acc2); },
            [](const SumStdDev& x, const SumStdDev& y) { return SumStdDev(x.acc1 + y.acc1, x.acc2 + y.acc2); });
        std = std::sqrt((total.acc1 - total.acc2 * total.acc2 / double(num_elems)) / double(num_elems));
    }

    return std;
}
}
