/**
* @file sourcefind.cpp
* Contains the prototypes and implementation of sourcefind functions
*/

#include "sourcefind.h"
#include "../common/vector_math.h"
#include <cassert>
#include <tbb/tbb.h>

namespace stp {

arma::uvec sigma_clip(arma::vec& data, double sigma, int iters)
{
    double median = arma::median(data);

    // Generate deviation data based on median function
    arma::vec deviation = data - median;
    arma::uvec valid_idx;
    double std = arma::stddev(deviation, 1);

    for (int i = 0; i < iters; i++) {
        double lower_sigma = -sigma * std;
        double upper_sigma = sigma * std;
        uint num_valid_elem = valid_idx.n_elem;

        valid_idx = arma::find((deviation > lower_sigma) && (deviation < upper_sigma));

        if (num_valid_elem == valid_idx.n_elem) {
            break;
        }
        std = arma::stddev(deviation(valid_idx), 1);
    }

    return valid_idx;
}

double estimate_rms(arma::mat& img, double sigma, uint iters)
{
    arma::vec data = arma::vectorise(img);
    assert(arma::is_finite(data)); // input data must have only finite values

    // The above sigma_clip function is not called, because we do not need the clipped vector data returned by sigma_clip
    // In the following, sigma_clip is combined with estimate_rms in order to save some computational complexity
    double median = arma::median(data);
    arma::vec deviation = data - median;

    double accu = vector_accumulate_parallel(deviation);
    uint valid_n_elem = deviation.n_elem;
    double std = vector_stddev_robust_parallel(deviation, accu / valid_n_elem);

    for (uint i = 0; i < iters; i++) {
        double lower_sigma = -sigma * std;
        double upper_sigma = sigma * std;

        tbb::combinable<double> local_accu(0.0);
        tbb::combinable<uint> rem_elem(0);

        tbb::parallel_for(tbb::blocked_range<size_t>(0, deviation.n_elem), [&deviation, &local_accu, &rem_elem, &valid_n_elem, &lower_sigma, &upper_sigma](const tbb::blocked_range<size_t>& r) {
            double tmp_accu = 0.0;
            uint tmp_rem_elem = 0;
            for (uint j = r.begin(); j != r.end(); j++) {
                double val = deviation[j];
                if (val < lower_sigma || val > upper_sigma) {
                    tmp_accu += val;
                    tmp_rem_elem++;
                    deviation[j] = arma::datum::nan;
                }
            }
            local_accu.local() += tmp_accu;
            rem_elem.local() += tmp_rem_elem;
        });

        uint total_rem_elem = rem_elem.combine([](uint x, uint y) { return x + y; });
        if (total_rem_elem == 0) {
            break;
        }

        valid_n_elem -= total_rem_elem;
        accu -= local_accu.combine([](double x, double y) { return x + y; });
        std = vector_stddev_robust_parallel(deviation, accu / valid_n_elem);
    }

    return std;
}

source_find_image::source_find_image(
    arma::mat input_data,
    double input_detection_n_sigma,
    double input_analysis_n_sigma,
    const std::experimental::optional<double>& input_rms_est,
    bool input_find_negative_sources)
    : data(input_data)
    , detection_n_sigma(input_detection_n_sigma)
    , analysis_n_sigma(input_analysis_n_sigma)
{
    rms_est = input_rms_est ? (*input_rms_est) : estimate_rms(data);

    bg_level = arma::median(arma::vectorise(input_data));

    _label_detection_islands_positive();

    if (input_find_negative_sources == true) {
        _label_detection_islands_negative();
    }

    assert(label_extrema.n_elem == label_extrema_label.n_elem);

    islands.resize(label_extrema.n_elem);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, label_extrema.n_elem), [this](const tbb::blocked_range<size_t>& r) {
        for (uint i = r.begin(); i != r.end(); i++) {
            islands[i] = std::move(island_params(data, label_map, label_extrema_label(i), label_extrema(i), label_extrema_linear_idx(i)));
        }
    });
}

arma::imat positive_comp(arma::mat& data, double analysis_thresh)
{
    arma::imat analysis_map(arma::size(data));
    tbb::parallel_for(tbb::blocked_range<size_t>(0, data.n_elem), [&data, &analysis_map, &analysis_thresh](const tbb::blocked_range<size_t>& r) {
        for (uint j = r.begin(); j != r.end(); j++) {
            if (data.at(j) > analysis_thresh) {
                analysis_map.at(j) = 1;
            } else {
                analysis_map.at(j) = 0;
            }
        }
    });
    return analysis_map;
}

arma::imat negative_comp(arma::mat& data, double analysis_thresh)
{
    arma::imat analysis_map(arma::size(data));
    tbb::parallel_for(tbb::blocked_range<size_t>(0, data.n_elem), [&data, &analysis_map, &analysis_thresh](const tbb::blocked_range<size_t>& r) {
        for (uint j = r.begin(); j != r.end(); j++) {
            if (data.at(j) < analysis_thresh) {
                analysis_map.at(j) = 1;
            } else {
                analysis_map.at(j) = 0;
            }
        }
    });
    return analysis_map;
}

std::pair<arma::vec, arma::uvec> positive_find_local_extrema(arma::mat& data, arma::imat& label_map, arma::sword n_labels)
{
    arma::vec data_maxs(n_labels);
    arma::uvec linear_idx(n_labels);

    const uint n_elem = data.n_elem;

    data_maxs.fill(std::numeric_limits<double>::min());

    for (uint i = 0; i < n_elem; i++) {
        double cmax, val;
        int label = label_map.at(i);

        if (label <= 0)
            continue;
        label -= 1;
        val = data.at(i);
        cmax = data_maxs.at(label);
        if (val > cmax) {
            data_maxs.at(label) = val;
            linear_idx.at(label) = i;
        }
    }

    return std::make_pair(std::move(data_maxs), std::move(linear_idx));
}

std::pair<arma::vec, arma::uvec> negative_find_local_extrema(arma::mat& data, arma::imat& label_map, arma::sword n_labels)
{
    arma::vec data_mins(n_labels);
    arma::uvec linear_idx(n_labels);
    const uint n_elem = data.n_elem;

    data_mins.fill(std::numeric_limits<double>::max());

    for (uint i = 0; i < n_elem; i++) {
        double cmin, val;
        int label = label_map.at(i);

        if (label <= 0)
            continue;
        label -= 1;
        val = data.at(i);
        cmin = data_mins.at(label);
        if (val < cmin) {
            data_mins.at(label) = val;
            linear_idx.at(label) = i;
        }
    }

    return std::make_pair(std::move(data_mins), std::move(linear_idx));
}

void source_find_image::_label_detection_islands_positive()
{
    double analysis_thresh_island = bg_level + analysis_n_sigma * rms_est;
    double detection_thresh_island = bg_level + detection_n_sigma * rms_est;

    arma::imat local_label_map(arma::size(data));
    arma::sword n_labels = labeling(positive_comp(data, analysis_thresh_island), local_label_map);
    std::pair<arma::vec, arma::uvec> all_label_extrema = positive_find_local_extrema(data, local_label_map, n_labels);

    assert(label_extrema.n_elem == label_extrema_label.n_elem);

    // Label islands that don't meet detection threshold
    arma::uvec label_extrema_idx_valid = arma::find(all_label_extrema.first > detection_thresh_island);
    arma::uvec label_extrema_idx_invalid = arma::find(all_label_extrema.first <= detection_thresh_island);

    uint prev_n_elem = label_extrema.n_elem;
    uint n_elem_valid = label_extrema_idx_valid.n_elem;
    uint n_elem_invalid = label_extrema_idx_invalid.n_elem;

    if (n_elem_valid > 0) {
        label_extrema.resize(prev_n_elem + n_elem_valid);
        label_extrema_linear_idx.resize(prev_n_elem + n_elem_valid);
        label_extrema_label.resize(prev_n_elem + n_elem_valid);

        label_extrema.subvec(prev_n_elem, arma::size(label_extrema_idx_valid)) = all_label_extrema.first.elem(label_extrema_idx_valid);
        label_extrema_linear_idx.subvec(prev_n_elem, arma::size(label_extrema_idx_valid)) = all_label_extrema.second.elem(label_extrema_idx_valid);
        label_extrema_idx_valid += 1;
        label_extrema_label.subvec(prev_n_elem, arma::size(label_extrema_idx_valid)) = arma::conv_to<arma::ivec>::from(label_extrema_idx_valid);
    }
    if (n_elem_invalid > 0) {
        label_extrema_idx_invalid.for_each([&local_label_map](arma::uvec::elem_type& val) { local_label_map.replace(val + 1, 0); });
    }
    label_map = local_label_map;
}

void source_find_image::_label_detection_islands_negative()
{
    double analysis_thresh_island = bg_level + (-1) * analysis_n_sigma * rms_est;
    double detection_thresh_island = bg_level + (-1) * detection_n_sigma * rms_est;

    arma::imat local_label_map(arma::size(data));
    arma::sword n_labels = labeling(negative_comp(data, analysis_thresh_island), local_label_map);
    std::pair<arma::vec, arma::uvec> all_label_extrema = negative_find_local_extrema(data, local_label_map, n_labels);

    assert(label_extrema.n_elem == label_extrema_label.n_elem);

    // Label islands that don't meet detection threshold
    arma::uvec label_extrema_idx_valid = arma::find(all_label_extrema.first < detection_thresh_island);
    arma::uvec label_extrema_idx_invalid = arma::find(all_label_extrema.first >= detection_thresh_island);

    uint prev_n_elem = label_extrema.n_elem;
    uint n_elem_valid = label_extrema_idx_valid.n_elem;
    uint n_elem_invalid = label_extrema_idx_invalid.n_elem;

    if (label_extrema_idx_valid.n_elem > 0) {
        label_extrema.resize(prev_n_elem + n_elem_valid);
        label_extrema_linear_idx.resize(prev_n_elem + n_elem_valid);
        label_extrema_label.resize(prev_n_elem + n_elem_valid);

        label_extrema.subvec(prev_n_elem, arma::size(label_extrema_idx_valid)) = all_label_extrema.first.elem(label_extrema_idx_valid);
        label_extrema_linear_idx.subvec(prev_n_elem, arma::size(label_extrema_idx_valid)) = all_label_extrema.second.elem(label_extrema_idx_valid);
        label_extrema_idx_valid += 1;
        label_extrema_label.subvec(prev_n_elem, arma::size(label_extrema_idx_valid)) = arma::conv_to<arma::ivec>::from(label_extrema_idx_valid) * (-1);
    }
    if (n_elem_invalid > 0) {
        label_extrema_idx_invalid.for_each([&local_label_map](arma::uvec::elem_type& val) { local_label_map.replace(val + 1, 0); });
    }
    // If extracting negative sources, flip the sign of the indices  and in the corresponding label map.
    label_map = label_map + (local_label_map * (-1));
}

island_params::island_params(
    arma::mat& input_data,
    arma::imat& label_map,
    int label,
    double l_extremum,
    uint l_extremum_linear_idx)
    : label_idx(label) // Index of region in label-map of source image.
    , extremum_val(l_extremum)
{

    // Determine if the label index is positive or negative
    const int l_sign = (label_idx < 0) ? -1 : 1;
    sign = l_sign;

    // Set indices of max/min sample
    arma::uvec sub = arma::ind2sub(arma::size(input_data), l_extremum_linear_idx);
    extremum_y_idx = sub(0);
    extremum_x_idx = sub(1);

    const uint data_ncols = input_data.n_cols;
    const uint data_nrows = input_data.n_rows;
    double b_x = 0.0;
    double b_y = 0.0;
    double b_sum = 0.0;

    // Analyses an 'island' to extract further parameters
    for (uint i = 0; i < data_ncols; i++) {
        for (uint j = 0; j < data_nrows; j++) {
            if (label_map.at(j, i) == label) {
                const double val = input_data.at(j, i);
                b_x += val * i;
                b_y += val * j;
                b_sum += val;
            }
        }
    }

    // Barycentric centre in x-pixel index
    xbar = b_x / b_sum;
    // Barycentric centre in y-pixel index
    ybar = b_y / b_sum;
}

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