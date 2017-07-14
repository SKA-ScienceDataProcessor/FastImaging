#include "matrix_math.h"
#include <cassert>
#include <cblas.h>
#include <math.h>
#include <thread>

#define BINMEDIAN_MIN_RANGE 0.0001
#define MIN_MEDSIZE 100

namespace stp {

double mat_median_exact(const arma::Mat<real_t>& data)
{
    double median = arma::datum::nan;
    size_t num_elems = data.n_elem;
    arma::Mat<real_t> auxdata = data;
    size_t k = (num_elems / 2); // half position

    // Use nth_element function to find the median
    arma::Mat<real_t>::iterator first = auxdata.begin();
    arma::Mat<real_t>::iterator last = auxdata.end();
    arma::Mat<real_t>::iterator nth = first + k; // Median position
    std::nth_element(first, nth, last);

    // If size is odd
    if (num_elems & 1) {
        median = (*nth); // Median
    }
    // If size is even
    else {
        // Find the second sample, from which median is interpolated
        arma::Mat<real_t>::iterator tfirst = auxdata.begin();
        arma::Mat<real_t>::iterator tlast = tfirst + k;
        const real_t val1 = (*nth);
        const real_t val2 = (*(std::max_element(tfirst, tlast)));
        median = (val1 + val2) / 2; // Median
    }

    return median;
}

DataStats mat_binmedian(const arma::Mat<real_t>& data)
{
    double median = arma::datum::nan;
    size_t num_elems = data.n_elem;
    const int N = 1000;

    // Compute mean and sigma quantities
    auto m_stats = mat_mean_and_stddev(data);
    double mean = m_stats.mean;
    double sigma = m_stats.sigma;

    // Use default nth_element algorithm if all elements are equal (sigma=0) or sigma is too small
    if ((sigma * 2) < BINMEDIAN_MIN_RANGE) {
        return DataStats(mean, sigma, mat_median_exact(data));
    }

    // Bin data across the interval [mean-sigma, mean+sigma]
    tbb::combinable<size_t> bottomcount_th(0);
    tbb::combinable<arma::uvec> bincounts_th(arma::uvec(N + 1).zeros());
    double scalefactor = (double)N / (2 * sigma);
    real_t leftend = mean - sigma;
    real_t rightend = mean + sigma;

    // Perform parallel binning
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elems, 4), [&](const tbb::blocked_range<size_t>& r) {
        size_t& l_bottomcount = bottomcount_th.local();
        arma::uvec& l_bincounts = bincounts_th.local();
        size_t i = r.begin();
        size_t rend = r.end();
        // Process 2 elements per loop iteration, only when range size is larger than 4
        // Benchmarks show that processing 2 elements per iteration is faster than processing 1 element
        if ((rend - i) > 4) {
            rend -= 2;
            for (; i < rend; i += 2) {
                const real_t& val = data[i];
                const real_t& val2 = data[i + 1];
                const int bin = (((val - leftend) * scalefactor));
                const int bin2 = (((val2 - leftend) * scalefactor));
                if (bin < 0) {
                    l_bottomcount++;
                } else {
                    if (bin <= N) {
                        l_bincounts[bin]++;
                    }
                }
                if (bin2 < 0) {
                    l_bottomcount++;
                } else {
                    if (bin2 <= N) {
                        l_bincounts[bin2]++;
                    }
                }
            }
            rend += 2;
        }
        // Process remaining elements
        for (; i < rend; i++) {
            const real_t& val = data[i];
            const int bin = (((val - leftend) * scalefactor));
            if (bin < 0) {
                l_bottomcount++;
            } else {
                if (bin <= N) {
                    l_bincounts[bin]++;
                }
            }
        }
    });

    // Combine temporary results from each thread
    size_t bottomcount = bottomcount_th.combine([](const size_t& x, const size_t& y) { return x + y; });
    arma::uvec bincounts = bincounts_th.combine([](const arma::uvec& x, const arma::uvec& y) { return x + y; });

    /*
     * Next steps of the algorithm to find Exact Median:
     * - apply sucessive binning
     * - copy samples from the selected bin
     * - apply nth_element on the copied samples
     */

    size_t k = (num_elems / 2) + 1; // half position + 1
    size_t oldmedbinsize = num_elems;
    size_t medbinsize = num_elems;
    double oldscalefactor;
    real_t oldleftend;
    int right_medbin = -1;
    int left_medbin = -1;

    // Recursive step for sucessive binning
    while (true) {
        // Find the bin that contains the median, and the order of the median within that bin
        bool found_bin = false;
        size_t count = bottomcount;
        k = (num_elems / 2) + 1; // half position + 1
        int i;
        for (i = 0; i < (N + 1); ++i) {
            count += bincounts[i];

            if (count >= k) {
                right_medbin = i;
                left_medbin = i;
                assert(k >= (count - bincounts[i]));
                k = k - (count - bincounts[i]);
                medbinsize = bincounts[i];
                found_bin = true; // Indicate that right_medbin was found
                assert(count >= bincounts[i]);
                break;
            }
        }

        // If (k-1) position does not belong to right_medbin (i.e. when k=1),
        // extend med bin with previous bins until new elements are added
        if (!(num_elems & 1)) { // k-1 position is needed only for even size
            if (found_bin && (k == 1)) {
                if (i > 0) { // Only makes sense if i > 0
                    i--;
                    assert(i <= N);
                    while (bincounts[i] == 0) {
                        i--;
                        if (i < 0) {
                            break;
                        }
                    }
                    if (i >= 0) {
                        k += bincounts[i];
                        medbinsize += bincounts[i];
                        left_medbin = i;
                    } else {
                        found_bin = false; // left_medbin was not found. Data binning must be stopped
                    }
                } else {
                    found_bin = false; // left_medbin was not found because i=0. Data binning must be stopped
                }
            }
        }

        // Update bin intervals
        if (found_bin) {
            oldscalefactor = scalefactor;
            oldleftend = leftend;
            leftend = (double)left_medbin / oldscalefactor + oldleftend; // right_medbin does not necessarily defines leftend for even data size, so left_medbin is used
            rightend = (double)(right_medbin + 1) / oldscalefactor + oldleftend;
            scalefactor = (double)N / (double)(rightend - leftend);
        } else {
            // Stop binning: right_medbin or left_medbin where not defined
            // If this is first iteration (right_medbin = -1), use nth_element function over all data
            if (right_medbin == -1) {
                return DataStats(mean, sigma, mat_median_exact(data));
            }
            // Otherwise, stop and use nth_element over the medbin data of the previous iteration
            break;
        }

        // Also stop if there's few points left in bin, bin sizes are too small, or medbin size didn't change
        if ((medbinsize < MIN_MEDSIZE) || (medbinsize < (num_elems / 10)) || ((rightend - leftend) < BINMEDIAN_MIN_RANGE) || (oldmedbinsize == medbinsize)) {
            oldmedbinsize = medbinsize; // oldmedbinsize is used after data binning
            break;
        }
        assert(right_medbin >= 0);

        oldmedbinsize = medbinsize;
        // Clear local thread temporaries
        bottomcount_th.clear();
        bincounts_th.clear();

        // Bin data across the new refined interval (parallel function)
        tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elems, 4), [&](const tbb::blocked_range<size_t>& r) {
            size_t& l_bottomcount = bottomcount_th.local();
            arma::uvec& l_bincounts = bincounts_th.local();
            size_t i = r.begin();
            size_t rend = r.end();
            // Process 2 elements per loop iteration, only when range size is larger than 4
            // Benchmarks show that processing 2 elements per iteration is faster than processing 1 element
            if ((rend - i) > 4) {
                rend -= 2;
                for (; i < rend; i += 2) {
                    const real_t& val = data[i];
                    const real_t& val2 = data[i + 1];
                    const int bin = (((val - leftend) * scalefactor));
                    const int bin2 = (((val2 - leftend) * scalefactor));
                    if (bin < 0) {
                        l_bottomcount++;
                    } else {
                        if (bin <= N) {
                            l_bincounts[bin]++;
                        }
                    }
                    if (bin2 < 0) {
                        l_bottomcount++;
                    } else {
                        if (bin2 <= N) {
                            l_bincounts[bin2]++;
                        }
                    }
                }
                rend += 2;
            }
            // Process remaining elements
            for (; i < rend; i++) {
                const real_t& val = data[i];
                const int bin = (((val - leftend) * scalefactor));
                if (bin < 0) {
                    l_bottomcount++;
                } else {
                    if (bin <= N) {
                        l_bincounts[bin]++;
                    }
                }
            }
        });

        // Combine temporary results from each thread
        bottomcount = bottomcount_th.combine([](const size_t& x, const size_t& y) { return x + y; });
        bincounts = bincounts_th.combine([](const arma::uvec& x, const arma::uvec& y) { return x + y; });
    }

    assert(medbinsize > 0);

    // Copy the selected bin samples to a new auxiliary buffer. Will be used with nth_element function
    tbb::concurrent_vector<real_t> auxdata;
    auxdata.reserve(oldmedbinsize);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elems, 4), [&](const tbb::blocked_range<size_t>& r) {
        size_t i = r.begin();
        size_t rend = r.end();
        // Process 2 elements per loop iteration, only when range size is larger than 4
        // Benchmarks show that processing 2 elements per iteration is faster than processing 1 element
        if ((rend - i) > 4) {
            rend -= 2;
            for (; i < rend; i += 2) {
                const real_t val = data[i];
                const real_t val2 = data[i + 1];
                const int bin = (((val - oldleftend) * oldscalefactor));
                const int bin2 = (((val2 - oldleftend) * oldscalefactor));
                if (bin >= left_medbin) {
                    if (bin <= right_medbin) {
                        auxdata.push_back(val);
                    }
                }
                if (bin2 >= left_medbin) {
                    if (bin2 <= right_medbin) {
                        auxdata.push_back(val2);
                    }
                }
            }
            rend += 2;
        }
        // Process remaining elements
        for (; i < rend; i++) {
            const real_t val = data[i];
            const int bin = (((val - oldleftend) * oldscalefactor));
            if (bin >= left_medbin) {
                if (bin <= right_medbin) {
                    auxdata.push_back(val);
                }
            }
        }
    });

    assert(oldmedbinsize == auxdata.size());
    assert((k - 1) < num_elems);

    // Use nth_element function to find the median from the remaining points
    tbb::concurrent_vector<real_t>::iterator first = auxdata.begin();
    tbb::concurrent_vector<real_t>::iterator last = auxdata.end();
    tbb::concurrent_vector<real_t>::iterator nth = first + k - 1; // Median position
    std::nth_element(first, nth, last);

    // If size is odd
    if (num_elems & 1) {
        median = (*nth); // Median
    }
    // If size is even
    else {
        // Find the second sample, from which median is interpolated
        tbb::concurrent_vector<real_t>::iterator tfirst = auxdata.begin();
        tbb::concurrent_vector<real_t>::iterator tlast = tfirst + k - 1;
        const real_t val1 = (*nth);
        const real_t val2 = (*(std::max_element(tfirst, tlast)));
        median = (val1 + val2) / 2; // Median
    }

    return DataStats(mean, sigma, median);
}

DataStats mat_median_binapprox(const arma::Mat<real_t>& data)
{
    double median = arma::datum::nan;
    size_t num_elems = data.n_elem;
    const int N = 1000;

    // Compute mean and sigma quantities
    auto m_stats = mat_mean_and_stddev(data);
    double mean = m_stats.mean;
    double sigma = m_stats.sigma;

    // Use default nth_element algorithm if all elements are equal (sigma=0) or sigma is too small
    if ((sigma * 2) < BINMEDIAN_MIN_RANGE) {
        return DataStats(mean, sigma, mat_median_exact(data));
    }

    // Bin data across the interval [mean-sigma, mean+sigma]
    tbb::combinable<size_t> bottomcount_th(0);
    tbb::combinable<arma::uvec> bincounts_th(arma::uvec(N + 1).zeros());
    const double scalefactor = (double)N / (2 * sigma);
    const real_t leftend = mean - sigma;

    // Perform parallel binning
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_elems, 4), [&](const tbb::blocked_range<size_t>& r) {
        size_t& l_bottomcount = bottomcount_th.local();
        arma::uvec& l_bincounts = bincounts_th.local();
        size_t i = r.begin();
        size_t rend = r.end();
        // Process 2 elements per loop iteration, only when range size is larger than 4
        // Benchmarks show that processing 2 elements per iteration is faster than processing 1 element
        if ((rend - i) > 4) {
            rend -= 2;
            for (; i < rend; i += 2) {
                const real_t& val = data[i];
                const real_t& val2 = data[i + 1];
                const int bin = ((val - leftend) * scalefactor);
                const int bin2 = ((val2 - leftend) * scalefactor);
                if (bin < 0) {
                    l_bottomcount++;
                } else {
                    if (bin <= N) {
                        l_bincounts[bin]++;
                    }
                }
                if (bin2 < 0) {
                    l_bottomcount++;
                } else {
                    if (bin2 <= N) {
                        l_bincounts[bin2]++;
                    }
                }
            }
            rend += 2;
        }
        // Process remaining elements
        for (; i < rend; i++) {
            const real_t& val = data[i];
            const int bin = ((val - leftend) * scalefactor);
            if (bin < 0) {
                l_bottomcount++;
            } else {
                if (bin <= N) {
                    l_bincounts[bin]++;
                }
            }
        }
    });

    size_t bottomcount = bottomcount_th.combine([](const size_t& x, const size_t& y) { return x + y; });
    arma::uvec bincounts = bincounts_th.combine([](const arma::uvec& x, const arma::uvec& y) { return x + y; });

    // If size is odd
    if (num_elems & 1) {
        // Find the bin that contains the median
        size_t k = (num_elems + 1) / 2;
        size_t count = bottomcount;

        for (uint i = 0; i < (N + 1); i++) {
            count += bincounts.at(i);

            if (count >= k) {
                median = double(i + 0.5) / scalefactor + leftend;
                break;
            }
        }
    }
    // If size is even
    else {
        // Find the bins that contains the medians
        // Median will be given by the average of two samples
        size_t k = num_elems / 2;
        size_t count = bottomcount;

        for (uint i = 0; i < (N + 1); i++) {
            count += bincounts.at(i);

            if (count >= k) {
                uint j = i; // This is right sample
                // Find left sample
                while (count == k) {
                    j++;
                    count += bincounts.at(j);
                    if (j == N) { // Ensure that j is not larger than N. In theory, this condition should never be true
                        break;
                    }
                }
                // Compute median from two samples
                median = double(i + j + 1) / (2 * scalefactor) + leftend;
                break;
            }
        }
    }
    // If the binapprox fails to find an approximate median, it falls back to the exact nth_element function
    if (!arma::is_finite(median)) {
        median = mat_median_exact(data);
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
            // Benchmarks show that processing 4 elements per iteration is faster than processing 1 element
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
            // Process remaining elements
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
