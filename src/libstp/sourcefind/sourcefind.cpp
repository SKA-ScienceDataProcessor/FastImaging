/**
* @file sourcefind.cpp
* Contains the prototypes and implementation of sourcefind functions
*/

#include "sourcefind.h"

double _estimate_rms(arma::mat img)
{
    return arma::stddev(vectorise(img), 1);
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
