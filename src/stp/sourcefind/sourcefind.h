/**
* @file sourcefind.h
* Contains the prototypes and implementation of sourcefind functions
*/

#ifndef SOURCE_FIND_H
#define SOURCE_FIND_H

#include "../common/ccl.h"
#include "../common/matrix_math.h"
#include "../common/matstp.h"
#include "../types.h"
#include <cassert>
#include <cfloat>
#include <functional>
#include <map>
#include <utility>

namespace stp {

extern std::vector<std::chrono::high_resolution_clock::time_point> times_sf;

/**
 * @brief Perform sigma-clip and estimate RMS of input matrix
 *
 * Compute Root mean square of input data after sigma-clipping (combines RMS estimation and sigma clip functions for improved computational performance).
 * Sigma clip is based on the pyhton's sigma_clip function in astropy.stats.
 *
 * @param[in] data (arma::Mat): Input data matrix. Data is not changed.
 * @param[in] sigma (double): The number of standard deviations to use for both the lower and upper clipping limit. Defaults to 3.
 * @param[in] iters (uint): The number of iterations for sigma clipping. Defaults to 5.
 * @param[in] stats (DataStats): Indicates the mean, sigma and median values to be used, if finite values are passed.
 *
 * @return (double): Computed Root Mean Square value.
 */
double estimate_rms(const arma::Mat<real_t>& data, double num_sigma = 3, uint iters = 5, DataStats stats = DataStats(arma::datum::nan, arma::datum::nan, arma::datum::nan));

/**
 * @brief island_params struct
 *
 * Data structure for representing source 'islands'
 *
 */
struct island_params {
    int label_idx;
    double extremum_val;
    int extremum_y_idx;
    int extremum_x_idx;
    double ybar;
    double xbar;
    int sign;

    island_params() = default;
    /**
     * @brief island_params constructor
     *
     * Initialized with parent image, label index, and peak-pixel value.
     *
     * @param[in] label (int): Index of region in label-map of source image.
     * @param[in] l_extremum (real_t): the extremum value
     * @param[in] l_extremum_coord_y (int): the y-coordinate index of the extremum value
     * @param[in] l_extremum_coord_x (int): the x-coordinate index of the extremum value
     * @param[in] barycentre_y (real_t): the y-value of barycentric centre
     * @param[in] barycentre_x (real_t): the x-value of barycentric centre
     */
    island_params(const int label, const real_t l_extremum, const int l_extremum_coord_y, const int l_extremum_coord_x, const real_t barycentre_y, const real_t barycentre_x);

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
    stp::MatStp<int> label_map;
    arma::Col<real_t> label_extrema_val_pos;
    arma::Col<real_t> label_extrema_val_neg;
    arma::uvec label_extrema_linear_idx_pos;
    arma::uvec label_extrema_linear_idx_neg;
    arma::ivec label_extrema_id_pos;
    arma::ivec label_extrema_id_neg;
    arma::mat label_extrema_barycentre_pos;
    arma::mat label_extrema_barycentre_neg;
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
     * @param[in] input_data (arma::Mat): Image data.
     * @param[in] detection_n_sigma (double): Detection threshold as multiple of RMS
     * @param[in] analysis_n_sigma (double): Analysis threshold as multiple of RMS
     * @param[in] rms_est (double): RMS estimate (may be 0.0, in which case RMS is estimated from the image data).
     * @param[in] find_negative_sources (bool): Find also negative sources (with signal is -1)
     * @param[in] sigmaclip_iters (uint): Number of iterations of sigma clip function.
     * @param[in] binapprox_median (bool): Compute approximated median using the fast binapprox method
     * @param[in] compute_bg_level (bool): Compute background level from median. If false, assumes bg_level = 0.
     * @param[in] compute_barycentre (bool): Compute barycentric centre of each island.
     */
    source_find_image(
        arma::Mat<real_t> input_data,
        double input_detection_n_sigma,
        double input_analysis_n_sigma,
        double input_rms_est = 0.0,
        bool find_negative_sources = true,
        uint sigmaclip_iters = 5,
        bool binapprox_median = false,
        bool compute_barycentre = true,
        bool generate_labelmap = true);

private:
    /** @brief _label_detection_islands function
     *
     *  Function to find connected regions which peak above or below a given threshold.
     *
     *  @return (uint) Number of valid labels
     */
    template <bool generateLabelMap>
    uint _label_detection_islands(bool find_negative_islands = true, bool computeBarycentre = true);
};
}

#endif /* SOURCE_FIND_H */
