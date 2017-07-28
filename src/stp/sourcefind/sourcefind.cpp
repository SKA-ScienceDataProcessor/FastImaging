/**
* @file sourcefind.cpp
* Contains the prototypes and implementation of sourcefind functions
*/

#include "sourcefind.h"
#include "../common/matrix_math.h"
#include "../common/matstp.h"
#include <cassert>
#include <tbb/tbb.h>

#define NUM_TIME_INST 10

namespace stp {

#ifdef FUNCTION_TIMINGS
std::vector<std::chrono::high_resolution_clock::time_point> times_iv;
std::vector<std::chrono::high_resolution_clock::time_point> times_sf;
#endif

// The sigma_clip step is combined with estimate_rms in order to save some computational complexity
// This is because we do not need the clipped vector data returned by sigma_clip
double estimate_rms(const arma::Mat<real_t>& data, double num_sigma, uint iters, DataStats stats)
{
    assert(arma::is_finite(data)); // input data must have only finite values

    // Compute mean, sigma and median if it was not received as input
    if (!arma::is_finite(stats.median)) {
        stats = mat_median_binapprox(data);
    }
    arma::uword n_elem = data.n_elem;
    real_t sigma = stats.sigma;
    const real_t median = stats.median;
    real_t mean = stats.mean - median;

    // Auxiliary variables storing the sum, squared sum and number of elements in the valid data
    DoublePair total_accu(double(mean) * double(n_elem), (double(sigma) * double(sigma) + double(mean) * double(mean)) * double(n_elem));
    size_t valid_n_elem = n_elem;

    // Represents a floation-point number that can be accessed using the integer type.
    FloatTwiddler prev_upper_sigma(std::numeric_limits<real_t>::max());

    // Perform sigma clip for the defined number of iterations
    for (uint i = 0; i < iters; i++) {
        // Represents a floation-point number that can be accessed using the integer type.
        FloatTwiddler upper_sigma(num_sigma * sigma);

        tbb::combinable<DoublePair> rem_accu(DoublePair(0.0, 0.0));
        tbb::combinable<size_t> rem_elem(0);

        assert((data.n_elem % 2) == 0);

        // Find outliers using a parallel loop
        tbb::parallel_for(tbb::blocked_range<size_t>(0, data.n_elem, 4), [&](const tbb::blocked_range<size_t>& r) {
            double& tmp_accu = rem_accu.local().d1;
            double& tmp_sqaccu = rem_accu.local().d2;
            size_t& tmp_rem_elem = rem_elem.local();
            size_t j = r.begin();
            size_t rend = r.end();
            // Process 2 elements per loop iteration, only when range size is larger than 4
            // Benchmarks show that processing 2 elements per iteration is faster than processing 1 element
            if ((rend - j) > 4) {
                rend -= 2;
                for (; j < rend; j += 2) {
                    const FloatTwiddler val(data[j] - median);
                    const FloatTwiddler val2(data[j + 1] - median);

// Replace the comparison of float types (commented 'if' lines below) by a faster method using integer comparison.
// Provided about 10% of speedup in estimate_rms function. This technique is described in: Optimizing software in C++, by Agner Fog.
// Section: 14.9 Using integer operations for manipulating floating point variables

//if (std::fabs(val) > upper_sigma) {
//    if (!(std::fabs(val) > prev_upper_sigma)) {
#ifdef USE_FLOAT
                    if ((val.i & 0x7FFFFFFF) > upper_sigma.i) {
                        if (!((val.i & 0x7FFFFFFF) > prev_upper_sigma.i)) {
#else
                        if ((val.i & 0x7FFFFFFFFFFFFFFF) > upper_sigma.i) {
                            if (!((val.i & 0x7FFFFFFFFFFFFFFF) > prev_upper_sigma.i)) {
#endif
                            tmp_accu += val.f;
                            tmp_sqaccu += (val.f * val.f);
                            tmp_rem_elem++;
                        }
                    }

//if (std::fabs(val2) > upper_sigma) {
//    if (!(std::fabs(val2) > prev_upper_sigma)) {
#ifdef USE_FLOAT
                    if ((val2.i & 0x7FFFFFFF) > upper_sigma.i) {
                        if (!((val2.i & 0x7FFFFFFF) > prev_upper_sigma.i)) {
#else
                        if ((val2.i & 0x7FFFFFFFFFFFFFFF) > upper_sigma.i) {
                            if (!((val2.i & 0x7FFFFFFFFFFFFFFF) > prev_upper_sigma.i)) {
#endif
                            tmp_accu += val2.f;
                            tmp_sqaccu += (val2.f * val2.f);
                            tmp_rem_elem++;
                        }
                    }
                }
                rend += 2;
            }

            // Process remaining elements
            for (; j < rend; j++) {
                const FloatTwiddler val(data[j] - median);

//if (std::fabs(val) > upper_sigma) {
//    if (!(std::fabs(val) > prev_upper_sigma)) {
#ifdef USE_FLOAT
                if ((val.i & 0x7FFFFFFF) > upper_sigma.i) {
                    if (!((val.i & 0x7FFFFFFF) > prev_upper_sigma.i)) {
#else
                        if ((val.i & 0x7FFFFFFFFFFFFFFF) > upper_sigma.i) {
                            if (!((val.i & 0x7FFFFFFFFFFFFFFF) > prev_upper_sigma.i)) {
#endif
                        tmp_accu += val.f;
                        tmp_sqaccu += (val.f * val.f);
                        tmp_rem_elem++;
                    }
                }
            }
        });

        // Count the removed elements and update the number of valid elements
        size_t total_rem_elem = rem_elem.combine([](size_t x, size_t y) { return x + y; });
        if (total_rem_elem == 0) {
            break;
        }
        valid_n_elem -= total_rem_elem;

        // Sum all the removed accu and sqaccu values
        auto tmpaccu = rem_accu.combine([](const DoublePair& x, const DoublePair& y) { return DoublePair(x.d1 + y.d1, x.d2 + y.d2); });
        // Subtract the accu and sqaccu values of the removed samples
        total_accu.d1 -= tmpaccu.d1;
        total_accu.d2 -= tmpaccu.d2;
        // Update sigma value
        sigma = std::sqrt(total_accu.d2 / double(valid_n_elem) - (total_accu.d1 / double(valid_n_elem)) * (total_accu.d1 / double(valid_n_elem)));
        prev_upper_sigma.f = upper_sigma.f;
    }

    return sigma;
}

source_find_image::source_find_image(
    const arma::Mat<real_t>& input_data,
    double input_detection_n_sigma,
    double input_analysis_n_sigma,
    double input_rms_est,
    bool find_negative_sources,
    uint sigma_clip_iters,
    bool binapprox_median,
    bool compute_barycentre,
    bool generate_labelmap)
    : detection_n_sigma(input_detection_n_sigma)
    , analysis_n_sigma(input_analysis_n_sigma)
{
#ifdef FUNCTION_TIMINGS
    times_sf.reserve(NUM_TIME_INST);
    times_sf.push_back(std::chrono::high_resolution_clock::now());
#endif

    // Compute statistics: mean, sigma, median
    DataStats data_stats(arma::datum::nan, arma::datum::nan, arma::datum::nan);
    if (binapprox_median) {
        data_stats = mat_median_binapprox(input_data);
        bg_level = data_stats.median;
    } else {
        data_stats = mat_binmedian(input_data);
        bg_level = data_stats.median;
    }

#ifdef FUNCTION_TIMINGS
    times_sf.push_back(std::chrono::high_resolution_clock::now());
#endif

    // Estimate RMS value, if rms_est is less or equal to 0.0
    rms_est = std::abs(input_rms_est) > 0.0 ? input_rms_est : estimate_rms(input_data, 3, sigma_clip_iters, data_stats);

#ifdef FUNCTION_TIMINGS
    times_sf.push_back(std::chrono::high_resolution_clock::now());
#endif

    // Perform label detection (for both positive and negative sources)
    uint numValidLabels = 0;
    if (generate_labelmap) {
        numValidLabels = _label_detection_islands<true>(input_data, find_negative_sources, compute_barycentre);
    } else {
        numValidLabels = _label_detection_islands<false>(input_data, find_negative_sources, compute_barycentre);
    }

#ifdef FUNCTION_TIMINGS
    times_sf.push_back(std::chrono::high_resolution_clock::now());
#endif

    assert(label_extrema_val_pos.n_elem == label_extrema_id_pos.n_elem);

    // Build vector of islands
    islands.reserve(numValidLabels);

    int h_shift = input_data.n_cols / 2;
    int v_shift = input_data.n_rows / 2;

    // Process positive islands
    for (size_t i = 0; i < label_extrema_id_pos.n_elem; i++) {
        if (label_extrema_id_pos.at(i)) {
            arma::uvec coord = arma::ind2sub(arma::size(input_data), label_extrema_linear_idx_pos.at(i));
#ifdef FFTSHIFT
            int y_idx = (int)coord[0];
            int x_idx = (int)coord[1];
#else
            // Shift coordinates because source find assumed input image was shifted
            int y_idx = (int)coord[0] < v_shift ? coord[0] + v_shift : (int)coord[0] - v_shift;
            int x_idx = (int)coord[1] < h_shift ? coord[1] + h_shift : (int)coord[1] - h_shift;
#endif
            islands.push_back(std::move(island_params(label_extrema_id_pos.at(i), label_extrema_val_pos.at(i), y_idx, x_idx, label_extrema_barycentre_pos.col(i)(0), label_extrema_barycentre_pos.col(i)(1))));
        }
    }

    // Process negative islands
    for (size_t i = 0; i < label_extrema_id_neg.n_elem; i++) {
        if (label_extrema_id_neg.at(i)) {
            arma::uvec coord = arma::ind2sub(arma::size(input_data), label_extrema_linear_idx_neg.at(i));
#ifdef FFTSHIFT
            int y_idx = (int)coord[0];
            int x_idx = (int)coord[1];
#else
            // Shift coordinates because source find assumed input image was shifted
            int y_idx = (int)coord[0] < v_shift ? coord[0] + v_shift : (int)coord[0] - v_shift;
            int x_idx = (int)coord[1] < h_shift ? coord[1] + h_shift : (int)coord[1] - h_shift;
#endif
            islands.push_back(std::move(island_params(label_extrema_id_neg.at(i), label_extrema_val_neg.at(i), y_idx, x_idx, label_extrema_barycentre_neg.col(i)(0), label_extrema_barycentre_neg.col(i)(1))));
        }
    }

    assert(islands.size() == numValidLabels);

#ifdef FUNCTION_TIMINGS
    times_sf.push_back(std::chrono::high_resolution_clock::now());
#endif
}

template <bool generateLabelMap>
uint source_find_image::_label_detection_islands(const arma::Mat<real_t>& data, bool find_negative_sources, bool computeBarycentre)
{
    // Compute analysis and detection thresholds
    const real_t analysis_thresh_pos = bg_level + analysis_n_sigma * rms_est;
    const real_t analysis_thresh_neg = bg_level - analysis_n_sigma * rms_est;
    const real_t detection_thresh_pos = bg_level + detection_n_sigma * rms_est;
    const real_t detection_thresh_neg = bg_level - detection_n_sigma * rms_est;
    std::tuple<MatStp<int>, MatStp<uint>, size_t, size_t> labeling_output;

    // Perform connected components labeling algorithm
    if (find_negative_sources) {
        labeling_output = labeling<true>(data, analysis_thresh_pos, analysis_thresh_neg);
    } else {
        labeling_output = labeling<false>(data, analysis_thresh_pos, analysis_thresh_neg);
    }

    // Process output of CCL function
    label_map = std::move(std::get<0>(labeling_output));
    MatStp<uint> P = std::move(std::get<1>(labeling_output));
    const uint num_l_pos = std::get<2>(labeling_output);
    const uint num_l_neg = std::get<3>(labeling_output);
    const uint* Pp = (uint*)P.colptr(0); // First column contains decision tree for positive labels.
    const uint* Pn = (uint*)P.colptr(1); // Second column contains decision tree for negative labels.

    assert(data.n_cols == label_map.n_cols);
    assert(data.n_rows == label_map.n_rows);

    // Arrays that store maximum and minimum extrema values.
    // These arrays are initialized with detection_thresh_pos and detection_thresh_neg values.
    // Updated labels will then present a value different than the initialized value.
    tbb::combinable<std::pair<arma::Col<real_t>, arma::uvec>> data_extrema_pos(std::make_pair(arma::Col<real_t>(num_l_pos).fill(detection_thresh_pos), arma::uvec(num_l_pos)));
    tbb::combinable<std::pair<arma::Col<real_t>, arma::uvec>> data_extrema_neg(std::make_pair(arma::Col<real_t>(num_l_neg).fill(detection_thresh_neg), arma::uvec(num_l_neg)));

    // Performs the final labeling stage, and searches the maximum/minimum sources (based on detection threshold).
    // These steps are merged in the same loop to minimize memory accesses.
    tbb::parallel_for(tbb::blocked_range<size_t>(0, data.n_elem), [&](const tbb::blocked_range<size_t>& r) {
        auto& r_data_extrema_pos = data_extrema_pos.local();
        auto& r_data_extrema_neg = data_extrema_neg.local();
        for (arma::uword i = r.begin(); i < r.end(); i++) {
            const int tmpL = label_map.at(i);
            if (tmpL == 0) {
                continue;
            }
            // tmpL is positive
            if (tmpL > 0) {
                int l = Pp[tmpL];
                if (l > 0) {
                    label_map.at(i) = l;
                    l--;
                    const real_t val = data.at(i);
                    const real_t cext = r_data_extrema_pos.first.at(l);
                    if (val > cext) {
                        r_data_extrema_pos.first.at(l) = val;
                        r_data_extrema_pos.second.at(l) = i;
                    }
                }
            } else {
                // tmpL is negative
                int l = Pn[-tmpL];
                if (l > 0) {
                    label_map.at(i) = -l;
                    l--;
                    const real_t val = data.at(i);
                    const real_t cext = r_data_extrema_neg.first.at(l);
                    if (val < cext) {
                        r_data_extrema_neg.first.at(l) = val;
                        r_data_extrema_neg.second.at(l) = i;
                    }
                }
            }
        }
    });

    // Combine maximum values of all threads
    std::pair<arma::Col<real_t>, arma::uvec>& comb_extrema_pos = data_extrema_pos.local();
    data_extrema_pos.combine_each([&](std::pair<arma::Col<real_t>, arma::uvec>& other) {
        for (size_t i = 0; i < comb_extrema_pos.first.n_elem; i++) {
            if (other.first.at(i) > comb_extrema_pos.first.at(i)) {
                comb_extrema_pos.first.at(i) = other.first.at(i);
                comb_extrema_pos.second.at(i) = other.second.at(i);
            }
        }
    });
    // Combine minimum values of all threads
    std::pair<arma::Col<real_t>, arma::uvec>& comb_extrema_neg = data_extrema_neg.local();
    data_extrema_neg.combine_each([&](std::pair<arma::Col<real_t>, arma::uvec>& other) {
        for (size_t i = 0; i < comb_extrema_neg.first.n_elem; i++) {
            if (other.first.at(i) < comb_extrema_neg.first.at(i)) {
                comb_extrema_neg.first.at(i) = other.first.at(i);
                comb_extrema_neg.second.at(i) = other.second.at(i);
            }
        }
    });

    // Create and init arrays related to label data
    size_t numValidLabels = num_l_pos + num_l_neg;
    label_extrema_val_pos = std::move(comb_extrema_pos.first);
    label_extrema_val_neg = std::move(comb_extrema_neg.first);
    label_extrema_linear_idx_pos = std::move(comb_extrema_pos.second);
    label_extrema_linear_idx_neg = std::move(comb_extrema_neg.second);
    label_extrema_id_pos.set_size(num_l_pos);
    label_extrema_id_neg.set_size(num_l_neg);
    label_extrema_barycentre_pos.set_size(2, num_l_pos);
    label_extrema_barycentre_neg.set_size(2, num_l_neg);
    if (!computeBarycentre) {
        label_extrema_barycentre_pos.zeros();
        label_extrema_barycentre_neg.zeros();
    }

    // Set label_extream_id for positive labels
    for (size_t i = 0; i < label_extrema_val_pos.n_elem; i++) {
        // Maximum extrema labels will be higher than detection_thresh_pos
        if (label_extrema_val_pos.at(i) > detection_thresh_pos) {
            label_extrema_id_pos.at(i) = i + 1;
        } else {
            label_extrema_id_pos.at(i) = 0;
            numValidLabels--;
        }
    }

    // Set label_extream_id for negative labels
    for (size_t i = 0; i < label_extrema_val_neg.n_elem; i++) {
        // Minimum extrema labels will be lower than detection_thresh_neg
        if (label_extrema_val_neg.at(i) < detection_thresh_neg) {
            label_extrema_id_neg.at(i) = -i - 1;
        } else {
            label_extrema_id_neg.at(i) = 0;
            numValidLabels--;
        }
    }

    // Compute barycentric centre or/and generate updated label map (i.e. remove undetected labels)
    if (computeBarycentre || generateLabelMap) {

        tbb::combinable<arma::Mat<double>> barycentre_pos(arma::Mat<double>(3, num_l_pos).zeros());
        tbb::combinable<arma::Mat<double>> barycentre_neg(arma::Mat<double>(3, num_l_neg).zeros());
        int h_shift = data.n_cols / 2;
        int v_shift = data.n_rows / 2;

        tbb::parallel_for(tbb::blocked_range<size_t>(0, data.n_cols), [&](const tbb::blocked_range<size_t>& r) {
            size_t li = arma::sub2ind(arma::size(data), 0, r.begin());
            arma::Mat<double>& r_barycentre_pos = barycentre_pos.local();
            arma::Mat<double>& r_barycentre_neg = barycentre_neg.local();
            for (arma::uword i = r.begin(); i < r.end(); i++) {
                for (arma::uword j = 0; j < data.n_rows; j++, li++) {
                    int label = label_map.at(li);
                    if (label == 0) {
                        continue;
                    }
                    int idx;
                    if (label > 0) {
                        idx = label - 1;
                        // This will remove weak sources (below the detection threshold)
                        if (label_extrema_id_pos.at(idx) == 0) {
                            // Update label_map with final label indexes (and clean weak sources)
                            if (generateLabelMap) {
                                label_map.at(li) = 0;
                            }
                            continue;
                        }
                    } else {
                        idx = -label - 1;
                        // This will remove weak sources (below the detection threshold)
                        if (label_extrema_id_neg.at(idx) == 0) {
                            // Update label_map with final label indexes (and clean weak sources)
                            if (generateLabelMap) {
                                label_map.at(li) = 0;
                            }
                            continue;
                        }
                    }
                    // Compute barycentre data
                    if (computeBarycentre) {
                        assert(idx > -1);
                        const double val = data.at(li);
#ifdef FFTSHIFT
                        const double y_idx = (int)j;
                        const double x_idx = (int)i;
#else
                        // Get shifted coordinates centered in the image
                        const double y_idx = (int)j < v_shift ? j + v_shift : (int)j - v_shift;
                        const double x_idx = (int)i < h_shift ? i + h_shift : (int)i - h_shift;
#endif
                        if (label > 0) {
                            r_barycentre_pos.at(0, idx) += y_idx * val;
                            r_barycentre_pos.at(1, idx) += x_idx * val;
                            r_barycentre_pos.at(2, idx) += val;
                        } else {
                            r_barycentre_neg.at(0, idx) += y_idx * val;
                            r_barycentre_neg.at(1, idx) += x_idx * val;
                            r_barycentre_neg.at(2, idx) += val;
                        }
                    }
                }
            }
        });

        // Sum computations of each thread and set label_extrema_barycentre arrays
        if (computeBarycentre) {
            auto all_barycentre_pos = barycentre_pos.combine([&](const arma::Mat<double>& x, const arma::Mat<double>& y) {
                auto r = x + y;
                return std::move(r);
            });
            auto all_barycentre_neg = barycentre_neg.combine([&](const arma::Mat<double>& x, const arma::Mat<double>& y) {
                auto r = x + y;
                return std::move(r);
            });

            for (arma::uword l = 0; l < all_barycentre_pos.n_cols; l++) {
                if (label_extrema_id_pos.at(l) != 0) {
                    label_extrema_barycentre_pos.at(0, l) = (double)all_barycentre_pos.at(0, l) / (double)all_barycentre_pos.at(2, l);
                    label_extrema_barycentre_pos.at(1, l) = (double)all_barycentre_pos.at(1, l) / (double)all_barycentre_pos.at(2, l);
                }
            }
            for (arma::uword l = 0; l < all_barycentre_neg.n_cols; l++) {
                if (label_extrema_id_neg.at(l) != 0) {
                    label_extrema_barycentre_neg.at(0, l) = (double)all_barycentre_neg.at(0, l) / (double)all_barycentre_neg.at(2, l);
                    label_extrema_barycentre_neg.at(1, l) = (double)all_barycentre_neg.at(1, l) / (double)all_barycentre_neg.at(2, l);
                }
            }
        }
    }
    return numValidLabels;
}

island_params::island_params(
    const int label,
    const real_t l_extremum,
    const int l_extremum_coord_y,
    const int l_extremum_coord_x,
    const real_t barycentre_y,
    const real_t barycentre_x)
    : label_idx(label) // Label index
    , extremum_val(l_extremum)
    , extremum_y_idx(l_extremum_coord_y)
    , extremum_x_idx(l_extremum_coord_x)
    , ybar(barycentre_y)
    , xbar(barycentre_x)
{

    // Determine if the label index is positive or negative
    sign = (label_idx < 0) ? -1 : 1;
}

// Compares two island objects
bool island_params::operator==(const island_params& other) const
{
    if (sign != other.sign) {
        return false;
    }
    if (xbar > other.xbar || xbar < other.xbar) {
        return false;
    }
    if (ybar > other.ybar || ybar < other.ybar) {
        return false;
    }
    if (extremum_x_idx > other.extremum_x_idx || extremum_x_idx < other.extremum_x_idx) {
        return false;
    }
    if (extremum_y_idx > other.extremum_y_idx || extremum_y_idx < other.extremum_y_idx) {
        return false;
    }
    if (extremum_val > other.extremum_val || extremum_val < other.extremum_val) {
        return false;
    }
    return true;
}
}
