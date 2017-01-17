/**
* @file sourcefind.h
* Contains the prototypes and implementation of sourcefind functions
*/

#ifndef SOURCE_FIND_H
#define SOURCE_FIND_H

#include "../common/ccl.h"
#include <cassert>
#include <cfloat>
#include <experimental/optional>
#include <functional>
#include <map>
#include <utility>

namespace stp {

/**
 * @brief sigma_clip function
 *
 * Perform sigma-clipping on the provided data.
 * Based on sigma_clip function of astropy.stats.
 *
 * @param[in] data (arma::vec): Input data to be sigma clipped.
 * @param[in] sigma (double): The number of standard deviations to use for both the lower and upper clipping limit. Defaults to 3.
 * @param[in] iters (int): The number of iterations to perform sigma clipping. Defaults to 5.
 *
 * @return (arma::uvec): An uvec array with the input data indexes accepted by the algorithm (i.e. not clipped indexes).
 *                       Indexes with non-finite data are also excluded.
*/
arma::uvec sigma_clip(arma::vec& data, double sigma = 3, int iters = 5);

/**
 * @brief estimate_rms function
 *
 * Compute Root mean square of input img.
 *
 * @param[in] img (arma::mat): Input array.
 *
 * @return (double): Root mean square value.
*/
double estimate_rms(arma::mat img);

/**
 * @brief positive_comp function
 *
 * Return the truth value of (data > analysis_thresh) for each element of data.
 *
 * @param[in] data (arma::mat): Array of values to be compared.
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
 * @param[in] data (arma::mat): Array of values to be compared.
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
 * @param[in] label_map (arma::mat): Array of integers representing different regions over which the maximum value of "data" is to be searched. Must have the same size as "data".
 * @param[in] data (arma::mat): Array of values. For each region specified by label_map, the maximum values of "data" over the region is computed.
 * @param[in] n_labels (arma::sword): Number of regions in label_map (does not count backgound - label 0).
 *
 * @return (arma::vec): Vector array with maximum values found for each label (vector index = label - 1).
*/
arma::vec positive_find_local_extrema(arma::mat& data, arma::imat& label_map, arma::sword n_labels);

/**
 * @brief negative_find_local_extrema function
 *
 * Find minimum values of an array over labeled regions
 *
 * @param[in] label_map (arma::mat): Array of integers representing different regions over which the minimum value of "data" is to be searched. Must have the same size as "data".
 * @param[in] data (arma::mat): Array of values. For each region specified by label_map, the minimum values of "data" over the region is computed.
 * @param[in] n_labels (int): Number of regions in label_map (does not count backgound - label 0).
 *
 * @return (arma::vec): Vector array with minimum values found for each label (vector index = label - 1).
*/
arma::vec negative_find_local_extrema(arma::mat& data, arma::imat& label_map, arma::sword n_labels);

struct island_params {
    arma::mat data;
    int label_idx;
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
     * @param[in] input_data (arma::mat): Image data.
     * @param[in] label (int): Index of region in label-map of source image.
     * @param[in] label_map (arma::imat): image representing connected components with label values
     * @param[in] l_extremum (double): the extremum value
     * @param[in] xgrid (arma::imat): X grid coordinates
     * @param[in] ygrid (arma::imat): Y grid coordinates
     *
    */
    island_params(
        arma::mat& input_data,
        int label,
        arma::imat& label_map,
        double l_extremum,
        arma::mat& xgrid,
        arma::mat& ygrid)
        : data(input_data)
        , extremum_val(l_extremum)
    {
        // Index of region in label-map of source image.
        label_idx = label;

        // Determine if the label index is positive or negative
        sign = (label_idx < 0) ? -1 : 1;

        // Analyses an 'island' to extract further parameters
        arma::uvec inv_positions = find(label_map != label_idx);
        data.elem(inv_positions).fill(0);

        // Barycentric centre in x-pixel index
        xbar = (arma::accu((xgrid % data) * sign) / (sign * arma::accu(data)));
        // Barycentric centre in y-pixel index
        ybar = (arma::accu((ygrid % data) * sign) / (sign * arma::accu(data)));

        // Set unwanted positions with Nan
        data.elem(inv_positions).fill(arma::datum::nan);

        // Returns max/min pixel index in np array ordering, i.e. (y_max, x_max)
        arma::uvec idx = (sign == 1) ? arma::ind2sub(arma::size(data), data.index_max()) : arma::ind2sub(arma::size(data), data.index_min());
        extremum_y_idx = idx.at(0);
        extremum_x_idx = idx.at(1);
    }

    /**
     * @brief Compare two island_params objects
     *
     * Compare if two island_params objects are exactly the same.
     *
     * @param[in] other (island_params): other island object to be compared.
     *
     * @return true/false
     */
    bool operator==(const island_params& other) const;
};

struct source_find_image {
    arma::mat data;
    arma::mat xgrid;
    arma::mat ygrid;
    arma::imat label_map;
    arma::vec label_extrema;
    arma::ivec label_extrema_idx;
    double detection_n_sigma;
    double analysis_n_sigma;
    double rms_est;
    double bg_level;
    double analysis_thresh;
    double detection_thresh;
    std::vector<island_params> islands;

public:
    source_find_image() = delete;

    /**
     * @brief SourceFindImage class
     *
     * Data structure for collecting intermediate results from source-extraction.
     *
     * This can be useful for verifying / debugging the sourcefinder results,
     * and intermediate results can also be reused to save recalculation.
     *
     * @param[in] data (arma::mat): Image data.
     * @param[in] detection_n_sigma (double): Detection threshold as multiple of RMS
     * @param[in] analysis_n_sigma (double): Analysis threshold as multiple of RMS
     * @param[in] rms_est (RMS_est): RMS estimate
     * @param[in] find_negative_sources (bool): Determine if the signal is -1 or 1 (negative/positive sources)
    */
    source_find_image(
        arma::mat input_data,
        double input_detection_n_sigma,
        double input_analysis_n_sigma,
        const std::experimental::optional<double>& input_rms_est = std::experimental::nullopt,
        bool input_find_negative_sources = true);

private:
    /** @brief _label_detection_islands function
    *
    *  Function (template + functor) to find connected regions which peak above/below a given threshold.
    *
    *  @param[in]: two template functions to proceed some calculations
    *
    *  @return No return, modify the object values
    */
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

        assert(label_extrema.n_elem == label_extrema_idx.n_elem);

        arma::uvec label_extrema_idx_neg = arma::find(all_label_extrema < detection_thresh_island);
        arma::uvec label_extrema_idx_pos = arma::find(all_label_extrema > detection_thresh_island);

        arma::uvec* le_idx;
        arma::uvec* le_idx_inv;

        if (comparison_op == negative_comp) {
            le_idx = &label_extrema_idx_neg;
            le_idx_inv = &label_extrema_idx_pos;
        } else {
            le_idx = &label_extrema_idx_pos;
            le_idx_inv = &label_extrema_idx_neg;
        }

        if (le_idx->n_elem) {
            int prev_n_elem = label_extrema.n_elem;

            label_extrema.resize(prev_n_elem + le_idx->n_elem);
            label_extrema_idx.resize(prev_n_elem + le_idx->n_elem);

            label_extrema.subvec(prev_n_elem, arma::size(*le_idx)) = all_label_extrema.elem(*le_idx);
            le_idx->for_each([](arma::uvec::elem_type& val) { val = (val + 1); });
            label_extrema_idx.subvec(prev_n_elem, arma::size(*le_idx)) = arma::conv_to<arma::ivec>::from(*le_idx) * sign;
        }
        le_idx_inv->for_each([&local_label_map](arma::uvec::elem_type& val) { local_label_map.replace(val + 1, 0); });

        if (comparison_op == negative_comp) {
            // If extracting negative sources, flip the sign of the indices  and in the corresponding label map.
            local_label_map = (-1) * local_label_map;
            label_map += local_label_map;
        } else {
            label_map = local_label_map;
        }
    }
};
}

#endif /* SOURCE_FIND_H */
