/**
* @file sourcefind.h
* Contains the prototypes and implementation of sourcefind functions
*/

#ifndef SOURCE_FIND_H
#define SOURCE_FIND_H

#include "../common/ccl.h"
#include "../types.h"
#include <cassert>
#include <cfloat>
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
 * @param[in] sigma (uint): The number of standard deviations to use for both the lower and upper clipping limit. Defaults to 3.
 * @param[in] iters (int): The number of iterations to perform sigma clipping. Defaults to 5.
 *
 * @return (arma::uvec): An uvec array with the input data indexes accepted by the algorithm (i.e. not clipped indexes).
 *                       Indexes with non-finite data are also excluded.
*/
arma::uvec sigma_clip(const arma::Col<real_t>& data, double sigma = 3, uint iters = 5);

/**
 * @brief Perform sigma-clip and estimate RMS of input matrix
 *
 * Compute Root mean square of input data after sigma-clipping (combines RMS and sigma clip for better computational performance).
 * Sigma clip is based on the sigma_clip function of astropy.stats.
 *
 * @param[in] data (arma::vec): Input array.
 * @param[in] sigma (double): The number of standard deviations to use for both the lower and upper clipping limit. Defaults to 3.
 * @param[in] iters (uint): The number of iterations to perform sigma clipping. Defaults to 5.
 *
 * @return (double): Root mean square value.
*/
double estimate_rms(const arma::Col<real_t>& data, double sigma = 3, uint iters = 5);

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
arma::Mat<char> positive_comp(const arma::Mat<real_t>& data, const double analysis_thresh);

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
arma::Mat<char> negative_comp(const arma::Mat<real_t>& data, const double analysis_thresh);

/**
 * @brief positive_find_local_extrema function
 *
 * Find maximum values of an array over labeled regions
 *
 * @param[in] label_map (arma::mat): Array of integers representing different regions over which the maximum value of "data" is to be searched. Must have the same size as "data".
 * @param[in] data (arma::mat): Array of values. For each region specified by label_map, the maximum values of "data" over the region is computed.
 * @param[in] n_labels (int): Number of regions in label_map (does not count backgound - label 0).
 *
 * @return (std::pair<arma::vec, arma::uvec>): Two vector arrays with maximum values found for each label (vector index = label - 1) and respective linear indices.
*/
std::pair<arma::Col<real_t>, arma::uvec> positive_find_local_extrema(const arma::Mat<real_t>& data, const arma::Mat<int>& label_map, int n_labels);

/**
 * @brief negative_find_local_extrema function
 *
 * Find minimum values of an array over labeled regions
 *
 * @param[in] label_map (arma::mat): Array of integers representing different regions over which the minimum value of "data" is to be searched. Must have the same size as "data".
 * @param[in] data (arma::mat): Array of values. For each region specified by label_map, the minimum values of "data" over the region is computed.
 * @param[in] n_labels (int): Number of regions in label_map (does not count backgound - label 0).
 *
 * @return (std::pair<arma::vec, arma::uvec>): Two vector arrays with minimum values found for each label (vector index = label - 1) and respective linear indices.
*/
std::pair<arma::Col<real_t>, arma::uvec> negative_find_local_extrema(const arma::Mat<real_t>& data, const arma::Mat<int>& label_map, int n_labels);

/**
 * @brief island_params struct
 *
 * Data structure for representing source 'islands'
 *
 */
struct island_params {
    int label_idx;
    double extremum_val;
    int extremum_x_idx;
    int extremum_y_idx;
    double xbar;
    double ybar;
    int sign;

    island_params() = default;
    /**
     * @brief island_params constructor
     *
     * Initialized with parent image, label index, and peak-pixel value.
     *
     * @param[in] input_data (arma::mat): Image data.
     * @param[in] label_map (arma::imat): image representing connected components with label values
     * @param[in] label (int): Index of region in label-map of source image.
     * @param[in] l_extremum (double): the extremum value
     * @param[in] l_extremum_linear_idx (double): the linear index of the extremum value
     * @param[in] compute_barycentre (bool): compute barycentric centre
     *
    */
    island_params(const arma::Mat<real_t>& input_data, const arma::Mat<int>& label_map, const int label, const double l_extremum, const uint l_extremum_linear_idx, const bool compute_barycentre);

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

/**
 * @brief SourceFindImage class
 *
 * Data structure for collecting intermediate results from source-extraction.
 *
 * This can be useful for verifying / debugging the sourcefinder results,
 * and intermediate results can also be reused to save recalculation.
 *
 */
class source_find_image {

public:
    arma::Mat<real_t> data;
    arma::Mat<int> label_map;
    arma::Col<real_t> label_extrema;
    arma::uvec label_extrema_linear_idx;
    arma::ivec label_extrema_label;
    double detection_n_sigma;
    double analysis_n_sigma;
    double rms_est;
    double bg_level;
    std::vector<island_params> islands;

    source_find_image() = delete;

    /**
     * @brief SourceFindImage constructor
     *
     * Constructs SourceFindImage structure and detects positive and negative (if input_find_negative_sources = true) sources
     *
     * @param[in] data (arma::mat): Image data.
     * @param[in] detection_n_sigma (double): Detection threshold as multiple of RMS
     * @param[in] analysis_n_sigma (double): Analysis threshold as multiple of RMS
     * @param[in] rms_est (RMS_est): RMS estimate (may be 0.0, in which case RMS is estimated from the image data).
     * @param[in] input_find_negative_sources (bool): Determine if the signal is -1 or 1 (negative/positive sources)
     * @param[in] sigmaclip_iters (uint): Number of iterations of sigma clip function.
     * @param[in] compute_bg_level (bool): Compute background level from median. If false, assumes bg_level = 0.
     * @param[in] compute_barycentre (bool): Compute barycentric centre of each island.
    */
    source_find_image(
        arma::Mat<real_t> input_data,
        double input_detection_n_sigma,
        double input_analysis_n_sigma,
        double input_rms_est = 0.0,
        bool input_find_negative_sources = true,
        uint sigmaclip_iters = 5,
        bool compute_bg_level = true,
        bool compute_barycentre = true);

private:
    /** @brief _label_detection_islands_positive function
    *
    *  Function to find connected regions which peak above a given threshold.
    *
    *  @return No return, modify the object values
    */
    void _label_detection_islands_positive();

    /** @brief _label_detection_islands_negative function
    *
    *  Function to find connected regions which peak below a given threshold.
    *
    *  @return No return, modify the object values
    */
    void _label_detection_islands_negative();
};
}

#endif /* SOURCE_FIND_H */
