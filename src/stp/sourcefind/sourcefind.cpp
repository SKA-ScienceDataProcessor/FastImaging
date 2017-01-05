/**
* @file sourcefind.cpp
* Contains the prototypes and implementation of sourcefind functions
*/

#include "sourcefind.h"
#include <cassert>

namespace stp {

arma::uvec sigma_clip(arma::vec& data, double sigma, int iters)
{
    arma::uvec valid_idx = find_finite(data);
    double median = arma::median(data.elem(valid_idx));

    // Generate deviation data based on median function
    arma::vec deviation = data;
    deviation.for_each([&median](arma::vec::elem_type& val) { val -= median; });

    arma::uvec invalid_idx_lower, invalid_idx_upper;
    double std;

    for (int i = 0; i < iters; i++) {

        std = arma::stddev(deviation.elem(valid_idx), 1);

        invalid_idx_lower = arma::find(deviation < (-sigma * std));
        deviation.elem(invalid_idx_lower).fill(arma::datum::nan);

        invalid_idx_upper = arma::find(deviation > (sigma * std));
        deviation.elem(invalid_idx_upper).fill(arma::datum::nan);

        valid_idx = find_finite(deviation);

        if ((invalid_idx_lower.n_elem == 0) && (invalid_idx_upper.n_elem == 0))
            break;
    }

    return valid_idx;
}

double estimate_rms(arma::mat img)
{
    arma::vec imgvec = arma::vectorise(img);

    return arma::stddev(imgvec.elem(sigma_clip(imgvec)), 1);
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
    , rms_est(input_rms_est ? (*input_rms_est) : estimate_rms(data))
    , bg_level(arma::mean(arma::mean(arma::real(input_data))))
    , analysis_thresh(0)
    , detection_thresh(0)
{
    xgrid = arma::mat(data.n_rows, data.n_cols);
    ygrid = arma::mat(data.n_rows, data.n_cols);

    arma::sword n = 0;
    xgrid.each_col([&n](arma::colvec& col) {
        col.fill(n);
        n++;
    });
    n = 0;
    ygrid.each_row([&n](arma::rowvec& row) {
        row.fill(n);
        n++;
    });

    _label_detection_islands(positive_comp, positive_find_local_extrema);

    if (input_find_negative_sources == true) {
        _label_detection_islands(negative_comp, negative_find_local_extrema);
    }

    assert(label_extrema.n_elem == label_extrema_idx.n_elem);

    for (uint i = 0; i < label_extrema.n_elem; i++) {
        islands.push_back(std::move(island_params(data, label_extrema_idx(i), label_map, label_extrema(i), xgrid, ygrid)));
    }
}

arma::imat positive_comp(arma::mat& data, double analysis_thresh)
{
    arma::imat analysis_map = arma::conv_to<arma::imat>::from(arma::zeros(data.n_rows, data.n_cols));
    analysis_map.elem(find(data > analysis_thresh)).ones();
    return analysis_map;
}

arma::imat negative_comp(arma::mat& data, double analysis_thresh)
{
    arma::imat analysis_map = arma::conv_to<arma::imat>::from(arma::zeros(data.n_rows, data.n_cols));
    analysis_map.elem(find(data < analysis_thresh)).ones();
    return analysis_map;
}

arma::vec positive_find_local_extrema(arma::mat& data, arma::imat& label_map, arma::sword n_labels)
{
    arma::vec labelmaxs(n_labels);
    double cmax, val;
    arma::sword index;

    arma::mat::iterator itd = data.begin();
    arma::imat::iterator itl = label_map.begin();

    labelmaxs.fill(DBL_MIN);

    for (; itd != data.end(); ++itd, ++itl) {
        if ((*itl) <= 0)
            continue;
        index = (*itl) - 1;
        val = (*itd);
        cmax = labelmaxs[index];
        if (val > cmax) {
            labelmaxs[index] = val;
        }
    }

    return labelmaxs;
}

arma::vec negative_find_local_extrema(arma::mat& data, arma::imat& label_map, arma::sword n_labels)
{
    arma::vec labelmins(n_labels);
    double cmin, val;
    int index;

    arma::mat::iterator itd = data.begin();
    arma::imat::iterator itl = label_map.begin();

    labelmins.fill(DBL_MAX);

    for (; itd != data.end(); ++itd, ++itl) {
        if ((*itl) <= 0)
            continue;
        index = (*itl) - 1;
        val = (*itd);
        cmin = labelmins[index];
        if (val < cmin) {
            labelmins[index] = val;
        }
    }

    return labelmins;
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
    if (extremum_val > other.extremum_val || extremum_y_idx < other.extremum_y_idx) {
        return false;
    }
    if (data.n_cols != other.data.n_cols || data.n_rows != other.data.n_rows) {
        return false;
    }
    // "A == B" returns a binary matrix indicating whether each entry is equal or not (it uses true or false)
    arma::mat d = data;
    arma::mat d_o = other.data;
    if (arma::accu(d.replace(arma::datum::nan, 0) == d_o.replace(arma::datum::nan, 0)) != data.n_elem) {
        return false;
    }

    return true;
}
}
