/**
* @file sourcefind.cpp
* @brief Implementation of classes and funtions of source find.
*/

#include "sourcefind.h"
#include "../common/ccl.h"
#include "../common/matrix_math.h"
#include "../global_macros.h"
#include "fitting.h"
#include <cassert>
#include <tbb/tbb.h>
#include <thread>

namespace stp {

std::vector<std::chrono::high_resolution_clock::time_point> times_sf;
std::vector<std::chrono::high_resolution_clock::time_point> times_ccl;

// The sigma_clip step is combined with estimate_rms in order to save some computational complexity
// This is because we do not need the clipped vector data returned by sigma_clip
double estimate_rms(const arma::Mat<real_t>& data, double num_sigma, uint iters, DataStats stats)
{
    assert(arma::is_finite(data)); // input data must have only finite values

    // Compute mean, sigma and median if it was not received as input
    if (!stats.median_valid) {
        stats = mat_binmedian(data);
    } else {
        if ((!stats.mean_valid) || (!stats.sigma_valid)) {
            DataStats tmp_stats = mat_mean_and_stddev(data);
            stats.mean = tmp_stats.mean;
            stats.mean_valid = true;
            stats.sigma = tmp_stats.sigma;
            stats.sigma_valid = true;
        }
    }

    arma::uword n_elem = data.n_elem;
    const real_t median = stats.median;
    // Subbtract median from mean
    real_t mean = stats.mean - median;
    // Sigma does not change when median is subtracted from data
    real_t sigma = stats.sigma;

    // Auxiliary variables storing the sum, squared sum and number of elements in the valid data
    DoublePair total_accu(double(mean) * double(n_elem), (double(sigma) * double(sigma) + double(mean) * double(mean)) * double(n_elem));
    size_t valid_n_elem = n_elem;

    // Represents a floation-point number that can be accessed using the integer type.
    FloatTwiddler prev_upper_sigma(std::numeric_limits<real_t>::max());

    // Perform Sigma-Clipping using the defined number of iterations
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

SourceFindImage::SourceFindImage(
    const arma::Mat<real_t>& input_data,
    double input_detection_n_sigma,
    double input_analysis_n_sigma,
    double input_rms_est,
    bool find_negative_sources,
    uint sigma_clip_iters,
    MedianMethod median_method,
    bool gaussian_fitting,
    bool ccl_4connectivity,
    bool generate_labelmap,
    int source_min_area,
    CeresDiffMethod ceres_diffmethod,
    CeresSolverType ceres_solvertype)
    : detection_n_sigma(input_detection_n_sigma)
    , analysis_n_sigma(input_analysis_n_sigma)
    , fit_gaussian(gaussian_fitting)
{
#ifdef FUNCTION_TIMINGS
    times_sf.reserve(NUM_TIME_INST);
    times_ccl.reserve(NUM_TIME_INST);
#endif
    TIMESTAMP_SOURCEFIND

    // Compute statistics: mean, sigma, median
    DataStats data_stats;
    switch (median_method) {
    case MedianMethod::ZEROMEDIAN:
        data_stats.median = 0.0;
        data_stats.median_valid = true;
        break;
    case MedianMethod::BINAPPROX:
        data_stats = mat_median_binapprox(input_data);
        break;
    case MedianMethod::BINMEDIAN:
        data_stats = mat_binmedian(input_data);
        break;
    case MedianMethod::NTHELEMENT:
        data_stats.median = mat_median_exact(input_data);
        data_stats.median_valid = true;
        break;
    }
    // Set background level
    bg_level = data_stats.median;

    STPLIB_DEBUG(spdlog::get("stplib"), "Sourcefind: Background level = {}", bg_level);

    TIMESTAMP_SOURCEFIND

    // Estimate RMS value, if rms_est is less or equal to 0.0
    rms_est = std::abs(input_rms_est) > 0.0 ? input_rms_est : estimate_rms(input_data, 3, sigma_clip_iters, data_stats);

    STPLIB_DEBUG(spdlog::get("stplib"), "Sourcefind: Estimated RMS value = {}", rms_est);

    TIMESTAMP_SOURCEFIND

    // Perform label detection (for both positive and negative sources)
    uint numValidLabels = 0;
    if (generate_labelmap) {
        numValidLabels = _label_detection_islands<true>(input_data, find_negative_sources, fit_gaussian, ccl_4connectivity);
    } else {
        numValidLabels = _label_detection_islands<false>(input_data, find_negative_sources, fit_gaussian, ccl_4connectivity);
    }

    STPLIB_DEBUG(spdlog::get("stplib"), "Sourcefind: Number of valid labels = {}", numValidLabels);

    TIMESTAMP_SOURCEFIND

    assert(label_extrema_val_pos.n_elem == label_extrema_id_pos.n_elem);

    // Build vector of islands
    islands.reserve(numValidLabels);

#ifndef FFTSHIFT
    int h_shift = input_data.n_cols / 2;
    int v_shift = input_data.n_rows / 2;
#endif

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
            if (label_extrema_numsamples_pos[i] >= source_min_area) {
                IslandParams island(label_extrema_id_pos[i], label_extrema_val_pos.at(i), y_idx, x_idx, label_extrema_numsamples_pos[i],
                    label_extrema_boundingbox_pos[i]);
                island.estimate_moments_fit(label_extrema_moments_pos.col(i)(0), label_extrema_moments_pos.col(i)(1),
                    label_extrema_moments_pos.col(i)(2), label_extrema_moments_pos.col(i)(3), label_extrema_moments_pos.col(i)(4),
                    rms_est, analysis_n_sigma);
                islands.push_back(std::move(island));
            }
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
            if (label_extrema_numsamples_neg[i] >= source_min_area) {
                IslandParams island(label_extrema_id_neg[i], label_extrema_val_neg.at(i), y_idx, x_idx, label_extrema_numsamples_neg[i],
                    label_extrema_boundingbox_neg[i]);
                island.estimate_moments_fit(label_extrema_moments_neg.col(i)(0), label_extrema_moments_neg.col(i)(1),
                    label_extrema_moments_neg.col(i)(2), label_extrema_moments_neg.col(i)(3), label_extrema_moments_neg.col(i)(4),
                    rms_est, analysis_n_sigma);
                islands.push_back(std::move(island));
            }
        }
    }

    // Perform gaussian fitting for each source
    if (fit_gaussian) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, islands.size()), [&](const tbb::blocked_range<size_t>& r) {
            const size_t& begin = r.begin();
            const size_t& end = r.end();
            for (size_t i = begin; i < end; i++) {
                islands[i].leastsq_fit_gaussian_2d(input_data, label_map, ceres_diffmethod, ceres_solvertype);
            }
        });
    }

    TIMESTAMP_SOURCEFIND
}

template <bool generateLabelMap>
uint SourceFindImage::_label_detection_islands(const arma::Mat<real_t>& data, bool find_negative_sources, bool gaussian_fitting, bool ccl_4connectivity)
{
    // Compute analysis and detection thresholds
    const real_t analysis_thresh_pos = bg_level + analysis_n_sigma * rms_est;
    const real_t analysis_thresh_neg = bg_level - analysis_n_sigma * rms_est;
    const real_t detection_thresh_pos = bg_level + detection_n_sigma * rms_est;
    const real_t detection_thresh_neg = bg_level - detection_n_sigma * rms_est;
    std::tuple<MatStp<int>, MatStp<uint>, size_t, size_t> labeling_output;

    TIMESTAMP_CCL

    // Perform connected components labeling algorithm
    if (ccl_4connectivity) {
        if (find_negative_sources) {
            labeling_output = labeling_4con<true>(data, analysis_thresh_pos, analysis_thresh_neg);
        } else {
            labeling_output = labeling_4con<false>(data, analysis_thresh_pos, analysis_thresh_neg);
        }
    } else {
        if (find_negative_sources) {
            labeling_output = labeling_8con<true>(data, analysis_thresh_pos, analysis_thresh_neg);
        } else {
            labeling_output = labeling_8con<false>(data, analysis_thresh_pos, analysis_thresh_neg);
        }
    }

    TIMESTAMP_CCL

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

    // Performs the final labeling stage and searches the maximum/minimum sources (based on detection threshold).
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

    TIMESTAMP_CCL

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
    label_extrema_moments_pos.set_size(5, num_l_pos);
    label_extrema_moments_neg.set_size(5, num_l_neg);
    label_extrema_boundingbox_pos.resize(num_l_pos);
    label_extrema_boundingbox_neg.resize(num_l_neg);

    // Set label_extream_id for valid positive labels. Set invalid sources with 0 label id
    for (size_t i = 0; i < label_extrema_val_pos.n_elem; i++) {
        // Maximum extrema labels will be higher than detection_thresh_pos
        if (label_extrema_val_pos.at(i) > detection_thresh_pos) {
            label_extrema_id_pos.at(i) = i + 1;
        } else {
            label_extrema_id_pos.at(i) = 0;
            numValidLabels--;
        }
    }

    // Set label_extream_id for negative labels. Set invalid sources with 0 label id
    for (size_t i = 0; i < label_extrema_val_neg.n_elem; i++) {
        // Minimum extrema labels will be lower than detection_thresh_neg
        if (label_extrema_val_neg.at(i) < detection_thresh_neg) {
            label_extrema_id_neg.at(i) = -i - 1;
        } else {
            label_extrema_id_neg.at(i) = 0;
            numValidLabels--;
        }
    }

    // The following code performs these steps:
    //  - Compute moments
    //  - Count number of samples of each island
    //  - Optional: Generate updated label map (i.e. remove undetected labels)
    //  - Optional: Find bounding box of each island
    tbb::combinable<arma::Mat<double>> moments_pos(arma::Mat<double>(6, num_l_pos).zeros());
    tbb::combinable<arma::Mat<double>> moments_neg(arma::Mat<double>(6, num_l_neg).zeros());
    tbb::combinable<arma::Col<int>> numsamples_pos(arma::Col<int>(num_l_pos).zeros());
    tbb::combinable<arma::Col<int>> numsamples_neg(arma::Col<int>(num_l_neg).zeros());

    // Stores the rectangular boxes defined around each source
    tbb::combinable<std::vector<BoundingBox>> boundingbox_pos(std::vector<BoundingBox>(num_l_pos, BoundingBox(data.n_rows, -1, data.n_cols, -1)));
    tbb::combinable<std::vector<BoundingBox>> boundingbox_neg(std::vector<BoundingBox>(num_l_neg, BoundingBox(data.n_rows, -1, data.n_cols, -1)));

#ifndef FFTSHIFT
    int h_shift = (int)(data.n_cols / 2);
    int v_shift = (int)(data.n_rows / 2);
#endif

    // Loop over image
    tbb::parallel_for(tbb::blocked_range<size_t>(0, data.n_cols), [&](const tbb::blocked_range<size_t>& r) {
        size_t li = arma::sub2ind(arma::size(data), 0, r.begin());
        arma::Mat<double>& r_moments_pos = moments_pos.local();
        arma::Mat<double>& r_moments_neg = moments_neg.local();
        arma::Col<int>& r_numsamples_pos = numsamples_pos.local();
        arma::Col<int>& r_numsamples_neg = numsamples_neg.local();
        std::vector<BoundingBox>& r_boundingbox_pos = boundingbox_pos.local();
        std::vector<BoundingBox>& r_boundingbox_neg = boundingbox_neg.local();

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

                // Calculate moments
                assert(idx > -1);
                double val = data.at(li);
#ifdef FFTSHIFT
                const double y_idx = (double)j;
                const double x_idx = (double)i;
#else
                // Get shifted coordinates centered in the image
                const double y_idx = (int)j < v_shift ? (double)(j + v_shift) : (double)(j - v_shift);
                const double x_idx = (int)i < h_shift ? (double)(i + h_shift) : (double)(i - h_shift);
#endif
                const double x_bar = x_idx * val;
                const double y_bar = y_idx * val;
                const double xx_bar = x_bar * x_idx;
                const double yy_bar = y_bar * y_idx;
                const double xy_bar = x_bar * y_idx;

                if (label > 0) {
                    r_numsamples_pos.at(idx)++;
                    r_moments_pos.at(0, idx) += x_bar;
                    r_moments_pos.at(1, idx) += y_bar;
                    r_moments_pos.at(2, idx) += xx_bar;
                    r_moments_pos.at(3, idx) += yy_bar;
                    r_moments_pos.at(4, idx) += xy_bar;
                    r_moments_pos.at(5, idx) += val;
                } else {
                    r_numsamples_neg.at(idx)++;
                    r_moments_neg.at(0, idx) += x_bar;
                    r_moments_neg.at(1, idx) += y_bar;
                    r_moments_neg.at(2, idx) += xx_bar;
                    r_moments_neg.at(3, idx) += yy_bar;
                    r_moments_neg.at(4, idx) += xy_bar;
                    r_moments_neg.at(5, idx) += val;
                }

                if (gaussian_fitting) {
#ifdef FFTSHIFT
                    const int col = (int)i;
                    const int row = (int)j;
#else
                    const int col = (int)i < h_shift ? (int)i + h_shift : (int)i - h_shift;
                    const int row = (int)j < v_shift ? (int)j + v_shift : (int)j - v_shift;
#endif
                    if (label > 0) {
                        if (col < r_boundingbox_pos[idx].left) {
                            r_boundingbox_pos[idx].left = col;
                        }
                        if (col > r_boundingbox_pos[idx].right) {
                            r_boundingbox_pos[idx].right = col;
                        }
                        if (row < r_boundingbox_pos[idx].top) {
                            r_boundingbox_pos[idx].top = row;
                        }
                        if (row > r_boundingbox_pos[idx].bottom) {
                            r_boundingbox_pos[idx].bottom = row;
                        }
                    } else {
                        if (col < r_boundingbox_neg[idx].left) {
                            r_boundingbox_neg[idx].left = col;
                        }
                        if (col > r_boundingbox_neg[idx].right) {
                            r_boundingbox_neg[idx].right = col;
                        }
                        if (row < r_boundingbox_neg[idx].top) {
                            r_boundingbox_neg[idx].top = row;
                        }
                        if (row > r_boundingbox_neg[idx].bottom) {
                            r_boundingbox_neg[idx].bottom = row;
                        }
                    }
                }
            }
        }
    });

    // Sum data of each thread to derive final moments
    auto all_moments_pos = moments_pos.combine([&](const arma::Mat<double>& x, const arma::Mat<double>& y) {
        auto r = x + y;
        return std::move(r);
    });
    auto all_moments_neg = moments_neg.combine([&](const arma::Mat<double>& x, const arma::Mat<double>& y) {
        auto r = x + y;
        return std::move(r);
    });

    // Sum num of samples counted by each thread
    label_extrema_numsamples_pos = numsamples_pos.combine([&](const arma::Col<int>& x, const arma::Col<int>& y) {
        auto r = x + y;
        return std::move(r);
    });
    label_extrema_numsamples_neg = numsamples_neg.combine([&](const arma::Col<int>& x, const arma::Col<int>& y) {
        auto r = x + y;
        return std::move(r);
    });

    // Set moments of positive sources in label_extrema_moments_pos array
    for (arma::uword l = 0; l < all_moments_pos.n_cols; l++) {
        // Skip invalid labels
        if (label_extrema_id_pos.at(l) != 0) {
            const double x_bar = all_moments_pos.at(0, l) / all_moments_pos.at(5, l);
            const double y_bar = all_moments_pos.at(1, l) / all_moments_pos.at(5, l);
            const double xx_bar = all_moments_pos.at(2, l) / all_moments_pos.at(5, l) - x_bar * x_bar;
            const double yy_bar = all_moments_pos.at(3, l) / all_moments_pos.at(5, l) - y_bar * y_bar;
            const double xy_bar = all_moments_pos.at(4, l) / all_moments_pos.at(5, l) - x_bar * y_bar;
            label_extrema_moments_pos.at(0, l) = x_bar;
            label_extrema_moments_pos.at(1, l) = y_bar;
            label_extrema_moments_pos.at(2, l) = xx_bar;
            label_extrema_moments_pos.at(3, l) = yy_bar;
            label_extrema_moments_pos.at(4, l) = xy_bar;
        }
    }
    // Set moments of negative sources in label_extrema_moments_neg array
    for (arma::uword l = 0; l < all_moments_neg.n_cols; l++) {
        // Skip invalid labels
        if (label_extrema_id_neg.at(l) != 0) {
            const double x_bar = all_moments_neg.at(0, l) / all_moments_neg.at(5, l);
            const double y_bar = all_moments_neg.at(1, l) / all_moments_neg.at(5, l);
            const double xx_bar = all_moments_neg.at(2, l) / all_moments_neg.at(5, l) - x_bar * x_bar;
            const double yy_bar = all_moments_neg.at(3, l) / all_moments_neg.at(5, l) - y_bar * y_bar;
            const double xy_bar = all_moments_neg.at(4, l) / all_moments_neg.at(5, l) - x_bar * y_bar;
            label_extrema_moments_neg.at(0, l) = x_bar;
            label_extrema_moments_neg.at(1, l) = y_bar;
            label_extrema_moments_neg.at(2, l) = xx_bar;
            label_extrema_moments_neg.at(3, l) = yy_bar;
            label_extrema_moments_neg.at(4, l) = xy_bar;
        }
    }

    // Combine the bounding box info of each thread
    if (gaussian_fitting) {
        label_extrema_boundingbox_pos.assign(num_l_pos, BoundingBox(data.n_rows, -1, data.n_cols, -1));
        boundingbox_pos.combine_each([&](const std::vector<BoundingBox>& bb) {
            for (size_t i = 0; i < bb.size(); i++) {
                if (bb[i].left < label_extrema_boundingbox_pos[i].left) {
                    label_extrema_boundingbox_pos[i].left = bb[i].left;
                }
                if (bb[i].right > label_extrema_boundingbox_pos[i].right) {
                    label_extrema_boundingbox_pos[i].right = bb[i].right;
                }
                if (bb[i].top < label_extrema_boundingbox_pos[i].top) {
                    label_extrema_boundingbox_pos[i].top = bb[i].top;
                }
                if (bb[i].bottom > label_extrema_boundingbox_pos[i].bottom) {
                    label_extrema_boundingbox_pos[i].bottom = bb[i].bottom;
                }
            }
        });
        label_extrema_boundingbox_neg.assign(num_l_neg, BoundingBox(data.n_rows, -1, data.n_cols, -1));
        boundingbox_neg.combine_each([&](const std::vector<BoundingBox>& bb) {
            for (size_t i = 0; i < bb.size(); i++) {
                if (bb[i].left < label_extrema_boundingbox_neg[i].left) {
                    label_extrema_boundingbox_neg[i].left = bb[i].left;
                }
                if (bb[i].right > label_extrema_boundingbox_neg[i].right) {
                    label_extrema_boundingbox_neg[i].right = bb[i].right;
                }
                if (bb[i].top < label_extrema_boundingbox_neg[i].top) {
                    label_extrema_boundingbox_neg[i].top = bb[i].top;
                }
                if (bb[i].bottom > label_extrema_boundingbox_neg[i].bottom) {
                    label_extrema_boundingbox_neg[i].bottom = bb[i].bottom;
                }
            }
        });
    }

    TIMESTAMP_CCL

    return numValidLabels;
}

IslandParams::IslandParams(
    const int label,
    const real_t l_extremum,
    const int l_extremum_coord_y,
    const int l_extremum_coord_x,
    const int l_num_samples,
    const BoundingBox& box)
    : label_idx(label) // Label index
    , extremum_val(l_extremum)
    , extremum_y_idx(l_extremum_coord_y)
    , extremum_x_idx(l_extremum_coord_x)
    , num_samples(l_num_samples)
    , bounding_box(box)
{
    // Determine if the label index is positive or negative
    sign = (label_idx < 0) ? -1 : 1;
}

void IslandParams::estimate_moments_fit(const double x_bar, const double y_bar, const double xx_bar, const double yy_bar, const double xy_bar, const double rms_est, const double analysis_n_sigma)
{
    const double working1 = (xx_bar + yy_bar) / 2.0;
    const double working2 = sqrt(((xx_bar - yy_bar) * (xx_bar - yy_bar) / 4) + xy_bar * xy_bar);
    const double trunc_semimajor_sq = working1 + working2;
    const double trunc_semiminor_sq = working1 - working2;

    // Semimajor / minor axes are under-estimated due to thresholding
    // Hanno calculated the following correction factor (eqns 2.60,2.61):
    const double pixel_threshold = analysis_n_sigma * rms_est;

    const double cutoff_ratio = sign * extremum_val / pixel_threshold;
    const double axes_scale_factor = 1.0 - log(cutoff_ratio) / (cutoff_ratio - 1.0);
    const double semimajor_est = sqrt(trunc_semimajor_sq / axes_scale_factor);
    const double semiminor_est = sqrt(trunc_semiminor_sq / axes_scale_factor);

    double theta_est = 0.5 * atan(2.0 * xy_bar / (xx_bar - yy_bar));

    if (theta_est * xy_bar < 0.0) {
        theta_est += arma::datum::pi / 2.0;
    }

    // Set gaussian parameters estimation
    moments_fit = Gaussian2dParams(extremum_val, x_bar, y_bar, semimajor_est, semiminor_est, theta_est);
    // Convert gaussian parameters to ensure that theta varies between -pi/2 and pi/2, and semimajor is larger than semiminor
    moments_fit.convert_to_constrained_parameters();
}

void IslandParams::leastsq_fit_gaussian_2d(const arma::Mat<real_t>& data, const arma::Mat<int>& label_map, CeresDiffMethod ceres_diffmethod, CeresSolverType ceres_solvertype)
{
    // Get the number of residuals
    int num_residuals = num_samples;

    // Return if it is one-pixel source
    if (num_residuals == 1) {
        leastsq_fit = Gaussian2dParams(extremum_val, 0.0, 0.0, 0.0, 0.0, 0.0);
        ceres_report.assign("ceres::Solve was not called.");
        return;
    }

    // The variable to solve for with its initial value.
    // It will be mutated in place by the solver.
    // Variable fields: amplitude, x0, y0, x_stddev, y_stddev and theta
    // Initial values are the given by the moments_fit
    double gaussian_params[] = { moments_fit.amplitude, moments_fit.x_centre, moments_fit.y_centre,
        moments_fit.semimajor, moments_fit.semiminor, moments_fit.theta };

    // Build the problem.
    ceres::Problem problem;

    // Set up the cost function (also known as residual).
    switch (ceres_diffmethod) {

    case CeresDiffMethod::AutoDiff: {
#ifndef FFTSHIFT
        int h_shift = (int)(data.n_cols / 2);
        int v_shift = (int)(data.n_rows / 2);
#endif
        // Compute each residual associated to label_idx
        for (int i = bounding_box.left; i <= bounding_box.right; ++i) {
            for (int j = bounding_box.top; j <= bounding_box.bottom; ++j) {
                const double x = double(i);
                const double y = double(j);
#ifdef FFTSHIFT
                const int& ii = i;
                const int& jj = j;
#else
                const int ii = i < h_shift ? i + h_shift : i - h_shift;
                const int jj = j < v_shift ? j + v_shift : j - v_shift;
#endif
                if (label_map.at(jj, ii) != label_idx) {
                    continue;
                }
                // This uses analytic derivatives
                ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<GaussianResidual, 1, 6>(
                    new GaussianResidual(data.at(jj, ii), x, y));
                problem.AddResidualBlock(cost_function, NULL, gaussian_params);
            }
        }
    } break;

    case CeresDiffMethod::AutoDiff_SingleResBlk: {
        // This uses auto-differentiation to obtain the derivative (jacobian).
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<GaussianAllResiduals, ceres::DYNAMIC, 6>(
            new GaussianAllResiduals(data, label_map, bounding_box, label_idx), num_residuals);
        problem.AddResidualBlock(cost_function, NULL, gaussian_params);
    } break;

    case CeresDiffMethod::AnalyticDiff: {
#ifndef FFTSHIFT
        int h_shift = (int)(data.n_cols / 2);
        int v_shift = (int)(data.n_rows / 2);
#endif
        // Compute each residual associated to label_idx
        for (int i = bounding_box.left; i <= bounding_box.right; ++i) {
            for (int j = bounding_box.top; j <= bounding_box.bottom; ++j) {
                const double x = double(i);
                const double y = double(j);
#ifdef FFTSHIFT
                const int& ii = i;
                const int& jj = j;
#else
                const int ii = i < h_shift ? i + h_shift : i - h_shift;
                const int jj = j < v_shift ? j + v_shift : j - v_shift;
#endif
                if (label_map.at(jj, ii) != label_idx) {
                    continue;
                }
                // This uses analytic derivatives
                ceres::CostFunction* cost_function = new GaussianAnalytic(data.at(jj, ii), x, y);
                problem.AddResidualBlock(cost_function, NULL, gaussian_params);
            }
        }
    } break;

    case CeresDiffMethod::AnalyticDiff_SingleResBlk: {
        // This uses analytic derivatives
        ceres::CostFunction* cost_function = new GaussianAnalyticAllResiduals(data, label_map, bounding_box, label_idx, num_residuals, 6);
        problem.AddResidualBlock(cost_function, NULL, gaussian_params);
    } break;

    default:
        assert(0);
        break;
    }

    // Solver options
    ceres::Solver::Options options;
    // Disable logging
    options.logging_type = ceres::SILENT;

    // Select solver type
    switch (ceres_solvertype) {

    case CeresSolverType::LinearSearch_BFGS: {
        options.minimizer_type = ceres::LINE_SEARCH;
        options.line_search_direction_type = ceres::BFGS;
    } break;

    case CeresSolverType::LinearSearch_LBFGS: {
        options.minimizer_type = ceres::LINE_SEARCH;
        options.line_search_direction_type = ceres::LBFGS;
    } break;

    case CeresSolverType::TrustRegion_DenseQR: {
        options.minimizer_type = ceres::TRUST_REGION;
        options.linear_solver_type = ceres::DENSE_QR;
    } break;

    default:
        assert(0);
        break;
    }

    // Run the solver
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Save the results
    leastsq_fit = Gaussian2dParams(gaussian_params[0], gaussian_params[1], gaussian_params[2], gaussian_params[3], gaussian_params[4], gaussian_params[5]);
    // Convert gaussian parameters to ensure that theta varies between -pi/2 and pi/2, and semimajor is larger than semiminor
    leastsq_fit.convert_to_constrained_parameters();
    assert(leastsq_fit.theta <= (arma::datum::pi / 2.0));
    assert(leastsq_fit.theta >= -(arma::datum::pi / 2.0));

    ceres_report.assign(summary.BriefReport());
    ceres_report.append(", Reason: " + summary.message);
}

// Compares two island objects
bool IslandParams::operator==(const IslandParams& other) const
{
    if (sign != other.sign) {
        return false;
    }
    if (extremum_x_idx != other.extremum_x_idx) {
        return false;
    }
    if (extremum_y_idx != other.extremum_y_idx) {
        return false;
    }
    if (abs(extremum_val - other.extremum_val) > fptolerance) {
        return false;
    }
    return true;
}
}
