/**
* @file sourcefind.h
* Contains the prototypes and implementation of sourcefind functions
*/

#ifndef SOURCE_FIND_H
#define SOURCE_FIND_H

#include "../common/ccl.h"
#include <cfloat>
#include <map>

const int positive_sign(1);
const int negative_sign(-1);
const double rms_est_false(-1000000000000000);

/**
 * @brief _estimate_rms function
 *
 * Compute Root mean square of input img.
 *
 * @param[in] img (array_like): Input array.
 *
 * @return (double): Root mean square value.
*/
double _estimate_rms(arma::mat img);

/**
 * @brief positive_comp function
 *
 * Return the truth value of (data > analysis_thresh) for each element of data.
 *
 * @param[in] data (array_like): Array of values to be compared.
 * @param[in] analysis_thresh (double): Threshold value used for comparison.
 *
 * @return (arma::imat): Binary array with "true" when (data > analysis_thresh) and "false" otherwise.
*/
arma::imat positive_comp(arma::mat& data, double analysis_thresh);

/**
 * @brief negative_comp function
 *
 * Return the truth value of (data < analysis_thresh) for each element of data.
 *
 * @param[in] data (array_like): Array of values to be compared.
 * @param[in] analysis_thresh (double): Threshold value used for comparison.
 *
 * @return (arma::imat): Binary array with "true" when (data < analysis_thresh) and "false" otherwise.
*/
arma::imat negative_comp(arma::mat& data, double analysis_thresh);

/**
 * @brief positive_find_local_extrema function
 *
 * Find maximum values of an array over labeled regions
 *
 * @param[in] label_map (array_like): Array of integers representing different regions over which the maximum value of "data" is to be searched. Must have the same size as "data".
 * @param[in] data (array_like): Array of values. For each region specified by label_map, the maximum values of "data" over the region is computed.
 * @param[in] n_labels (int): Number of regions in label_map (does not count backgound - label 0).
 *
 * @return (arma::vec): Vector array with maximum values found for each label (vector index = label - 1).
*/
arma::vec positive_find_local_extrema(arma::mat& data, arma::imat& label_map, arma::sword n_labels);

/**
 * @brief negative_find_local_extrema function
 *
 * Find minimum values of an array over labeled regions
 *
 * @param[in] label_map (array_like): Array of integers representing different regions over which the minimum value of "data" is to be searched. Must have the same size as "data".
 * @param[in] data (array_like): Array of values. For each region specified by label_map, the minimum values of "data" over the region is computed.
 * @param[in] n_labels (int): Number of regions in label_map (does not count backgound - label 0).
 *
 * @return (arma::vec): Vector array with minimum values found for each label (vector index = label - 1).
*/
arma::vec negative_find_local_extrema(arma::mat& data, arma::imat& label_map, arma::sword n_labels);

struct island_params {
    arma::mat data;
    double extremum_val;
    double extremum_x_idx;
    double extremum_y_idx;
    double xbar;
    double ybar;
    int sign;

public:
    island_params() = default;
    /**
     * @brief island_params constructor
     *
     * Data structure for representing source 'islands'
     * Initialized with parent image, label index, and peak-pixel value.
     *
     * @param[in] label_idx (int): Index of region in label-map of source image.
     * @param[in] peak_val (float): Peak pixel value
     * @param[in] peak_x_idx (int): Peak pixel x-index
     * @param[in] peak_y_idx (int): Peak pixel y-index
     * @param[in] xbar (float): Barycentric centre in x-pixel index
     *
    */
    island_params(
        arma::mat& input_data,
        int label_idx,
        int input_sign, //_positive_negative_sign_validator: inside -1 to 1
        arma::imat& label_map,
        double l_extremum,
        arma::mat& xgrid,
        arma::mat& ygrid)
        : data(input_data)
        , extremum_val(l_extremum)
        , sign(input_sign)
    {
        // Analyses an 'island' to extract further parameters OLD: _label_mask
        data.elem(find(label_map != label_idx)).fill(arma::datum::nan);
        arma::mat data_aux = data;
        data_aux.replace(arma::datum::nan, 0);

        // Returns max/min pixel index in np array ordering, i.e. (y_max, x_max) OLD: _extremum_pixel_index
        arma::uvec idx = (sign == positive_sign) ? arma::ind2sub(arma::size(data), data.index_max()) : arma::ind2sub(arma::size(data), data.index_min());
        extremum_y_idx = idx.at(0);
        extremum_x_idx = idx.at(1);
        xbar = (arma::accu((xgrid % data_aux) * sign) / (sign * arma::accu(data_aux)));
        ybar = (arma::accu((ygrid % data_aux) * sign) / (sign * arma::accu(data_aux)));
    }

    bool operator==(const island_params& other) const;
};

struct source_find_image {
    arma::mat data;
    arma::mat xgrid;
    arma::mat ygrid;
    arma::imat label_map;
    std::map<int, double> label_extrema;
    double detection_n_sigma;
    double analysis_n_sigma;
    double rms_est;
    double bg_level;
    double analysis_thresh;
    double detection_thresh;
    std::vector<island_params> islands;

public:
    /**
     * @brief SourceFindImage class
     *
     * Data structure for collecting intermediate results from source-extraction.
     *
     * This can be useful for verifying / debugging the sourcefinder results,
     * and intermediate results can also be reused to save recalculation.
     *
     * @param[in] data (array_like): numpy.ndarray or numpy.ma.MaskedArray containing image data.
     * @param[in] detection_n_sigma (float): Detection threshold as multiple of RMS
     * @param[in] analysis_n_sigma (float): Analysis threshold as multiple of RMS
     * @param[in] rms_est (float): RMS estimate (may be `None`, in which case RMS is estimated from the image data via sigma-clipping).
     *
    */
    source_find_image() = default;
    source_find_image(
        arma::mat input_data,
        double input_detection_n_sigma,
        double input_analysis_n_sigma,
        double input_rms_est,
        bool input_find_negative_sources)
        : data(input_data)
        , detection_n_sigma(input_detection_n_sigma)
        , analysis_n_sigma(input_analysis_n_sigma)
        , rms_est((input_rms_est < rms_est_false) ? _estimate_rms(data) : input_rms_est)
        , bg_level(arma::mean(arma::mean(arma::real(input_data))))
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
        for (std::map<int, double>::iterator it = label_extrema.begin(); it != label_extrema.end(); ++it) {
            // Determine if the label index is positive or negative
            int l_idx = std::get<0>(*it);
            double l_extremum = std::get<1>(*it);
            int l_sign = int(copysign(1, l_idx));
            islands.push_back(island_params(data, l_idx, l_sign, label_map, l_extremum, xgrid, ygrid));
        }
    }

    template <typename T, typename A>
    void _label_detection_islands(const T& comparison_op, const A& find_local_extrema)
    {
        arma::imat local_label_map;
        local_label_map.zeros(data.n_rows, data.n_cols);
        double sign = (comparison_op == negative_comp) ? -1 : 1;
        double analysis_thresh_island = bg_level + sign * analysis_n_sigma * rms_est;
        double detection_thresh_island = bg_level + sign * detection_n_sigma * rms_est;

        arma::sword n_labels = labeling(comparison_op(data, analysis_thresh_island), local_label_map);
        arma::vec all_label_extrema = find_local_extrema(data, local_label_map, n_labels);

        if (comparison_op == negative_comp) {

            // Label islands that don't meet detection threshold
            for (arma::uword zero_idx(0); zero_idx < all_label_extrema.n_elem; ++zero_idx) {
                arma::uword label = zero_idx + 1;

                if (all_label_extrema[zero_idx] < detection_thresh_island) {
                    label_extrema.insert(std::pair<int, double>(label * (-1), all_label_extrema[zero_idx]));
                } else {
                    local_label_map.replace(label, 0);
                }
            }

            // If extracting negative sources, flip the sign of the indices  and in the corresponding label map.
            local_label_map = negative_sign * local_label_map;
            label_map += local_label_map;

        } else {

            // Label islands that don't meet detection threshold
            for (arma::uword zero_idx(0); zero_idx < all_label_extrema.n_elem; ++zero_idx) {
                arma::uword label = zero_idx + 1;

                if (all_label_extrema[zero_idx] > detection_thresh_island) {
                    label_extrema.insert(std::pair<int, double>(label, all_label_extrema[zero_idx]));
                } else {
                    local_label_map.replace(label, 0);
                }
            }

            label_map = local_label_map;
        }
    }
};

#endif /* SOURCE_FIND_H */
