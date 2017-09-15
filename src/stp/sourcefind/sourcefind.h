/**
* @file sourcefind.h
* @brief Classes and funtion prototypes of source find.
*/

#ifndef SOURCE_FIND_H
#define SOURCE_FIND_H

#include "../common/ccl.h"
#include "../common/matrix_math.h"
#include "../common/matstp.h"
#include "../types.h"
#include "fitting.h"
#include <cassert>
#include <cfloat>
#include <functional>
#include <map>
#include <utility>

namespace stp {

extern std::vector<std::chrono::high_resolution_clock::time_point> times_sf;

/**
 * @brief Represents a floation-point number that can be accessed using the integer type.
 *        This union allows to perform bitwise operations using the integer type interface.
 */
union FloatTwiddler {
public:
    FloatTwiddler(real_t num = 0.0f)
        : f(num)
    {
    }

    real_t f;
#ifdef USE_FLOAT
    int32_t i;
#else
    int64_t i;
#endif
};

/**
 * @brief Perform sigma-clip and estimate RMS of input matrix
 *
 * Compute Root mean square of input data after sigma-clipping (combines RMS estimation and sigma clip functions for improved computational performance).
 * Sigma clip is based on the pyhton's sigma_clip function in astropy.stats.
 *
 * @param[in] data (arma::Mat): Input data matrix. Data is not changed.
 * @param[in] sigma (double): The number of standard deviations to use for both the lower and upper clipping limit. Defaults to 3.
 * @param[in] iters (uint): The number of iterations for sigma clipping. Defaults to 5.
 * @param[in] stats (DataStats): The mean, sigma and median values to be used. If non-finite values are passed, they must be computed.
 *
 * @return (double): Computed Root Mean Square value.
 */
double estimate_rms(const arma::Mat<real_t>& data, double num_sigma = 3, uint iters = 5, DataStats stats = DataStats(arma::datum::nan, arma::datum::nan, arma::datum::nan));

/**
 * @brief IslandParams struct
 *
 * Data structure for representing source 'islands'
 *
 */
struct IslandParams {
    int label_idx;
    double extremum_val;
    int extremum_y_idx;
    int extremum_x_idx;
    int sign;
    int num_samples;
    BoundingBox bounding_box;
    Gaussian2dParams moments_fit;
    Gaussian2dParams leastsq_fit;
    std::string ceres_report;

    /**
     * @brief IslandParams default constructor
     */
    IslandParams() = default;

    /**
     * @brief IslandParams constructor
     *
     * Initialized with label index, and peak-pixel value, peak-pixel position and barycentre.
     *
     * @param[in] label (int): Index of region in label-map of source image
     * @param[in] l_extremum (real_t): the extremum value
     * @param[in] l_extremum_coord_y (int): the y-coordinate index of the extremum value
     * @param[in] l_extremum_coord_x (int): the x-coordinate index of the extremum value
     * @param[in] l_num_samples (int): number of samples in the island
     * @param[in] bbox (BoundingBox): the bounding box defined around the source
     */
    IslandParams(const int label, const real_t l_extremum, const int l_extremum_coord_y,
        const int l_extremum_coord_x, const int l_num_samples, const BoundingBox& box = BoundingBox());

    /**
     * @brief Estimate initial 2D gaussian fit to the island using the method of moments.
     *
     * The moments information used to estimate the 2D gaussian are passed as input parameters.
     *
     * @param[in] x_bar (double): First moment - x barycentre
     * @param[in] y_bar (double): First moment - y barycentre
     * @param[in] xx_bar (double): Second moment - xx
     * @param[in] yy_bar (double): Second moment - yy
     * @param[in] xy_bar (double): Second moment - xy
     * @param[in] rms_est (double): RMS estimation
     * @param[in] analysis_n_sigma (double): Analysis threshold as multiple of RMS
     */
    void estimate_moments_fit(const double x_bar, const double y_bar, const double xx_bar, const double yy_bar, const double xy_bar, const double rms_est, const double analysis_n_sigma);

    /**
     * @brief Fit 2D gaussian to the island.
     *
     * Fit 2D gaussian to the island using non-linear least-squares optimisation methods implemented by ceres library.
     * Requires image and label map information.
     *
     * @param[in] data (arma::Mat<real_t>): Image data matrix
     * @param[in] label_map (arma::Mat<int>): Label map matrix
     * @param[in] ceres_diffmethod (CeresDiffMethod): Differentiation method used by ceres library for gaussian fitting.
     * @param[in] ceres_solvertype (CeresSolverType): Solver type used by ceres library for gaussian fitting.
     */
    void leastsq_fit_gaussian_2d(const arma::Mat<real_t>& data, const arma::Mat<int>& label_map, CeresDiffMethod ceres_diffmethod, CeresSolverType ceres_solvertype);

    /**
     * @brief Compare two IslandParams objects
     *
     * Compare if two IslandParams objects are exactly the same.
     *
     * @param[in] other (IslandParams): other island object to be compared.
     *
     * @return (bool )true/false
     */
    bool operator==(const IslandParams& other) const;
};

/**
 * @brief The SourceFindImage class for source detection.
 *
 * The structure collects intermediate results from source-detection.
 * This can be useful for verifying / debugging the sourcefinder results,
 * and intermediate results can also be reused to save recalculation.
 *
 */
class SourceFindImage {

public:
    MatStp<int> label_map;
    arma::Col<real_t> label_extrema_val_pos;
    arma::Col<real_t> label_extrema_val_neg;
    arma::uvec label_extrema_linear_idx_pos;
    arma::uvec label_extrema_linear_idx_neg;
    arma::ivec label_extrema_id_pos;
    arma::ivec label_extrema_id_neg;
    arma::mat label_extrema_moments_pos;
    arma::mat label_extrema_moments_neg;
    arma::Col<int> label_extrema_numsamples_pos;
    arma::Col<int> label_extrema_numsamples_neg;
    tbb::concurrent_vector<BoundingBox> label_extrema_boundingbox_pos;
    tbb::concurrent_vector<BoundingBox> label_extrema_boundingbox_neg;
    double detection_n_sigma;
    double analysis_n_sigma;
    double rms_est;
    double bg_level;
    std::vector<IslandParams> islands;
    bool fit_gaussian;

    // There's no default constructor
    SourceFindImage() = delete;

    /**
     * @brief SourceFindImage constructor
     *
     * Constructs SourceFindImage structure and detects positive and negative (if input_find_negative_sources = true) sources
     *
     * @param[in] input_data (arma::Mat): Image data.
     * @param[in] input_detection_n_sigma (double): Detection threshold as multiple of RMS
     * @param[in] input_analysis_n_sigma (double): Analysis threshold as multiple of RMS
     * @param[in] input_rms_est (double): RMS estimate (may be 0.0, in which case RMS is estimated from the image data).
     * @param[in] find_negative_sources (bool): Find also negative sources (with signal is -1)
     * @param[in] sigmaclip_iters (uint): Number of iterations of sigma clip function.
     * @param[in] binapprox_median (bool): Compute approximated median using the fast binapprox method
     * @param[in] gaussian_fitting (bool): Perform gaussian fitting for each island.
     * @param[in] generate_labelmap (bool): Update the final label map by removing the sources below the detection threshold.
     * @param[in] ceres_diffmethod (CeresDiffMethod): Differentiation method used by ceres library for gaussian fitting.
     * @param[in] ceres_solvertype (CeresSolverType): Solver type used by ceres library for gaussian fitting.
     */
    SourceFindImage(
        const arma::Mat<real_t>& input_data,
        double input_detection_n_sigma,
        double input_analysis_n_sigma,
        double input_rms_est = 0.0,
        bool find_negative_sources = true,
        uint sigma_clip_iters = 5,
        bool binapprox_median = false,
        bool gaussian_fitting = false,
        bool generate_labelmap = true,
        CeresDiffMethod ceres_diffmethod = CeresDiffMethod::AnalyticDiff_SingleResBlk,
        CeresSolverType ceres_solvertype = CeresSolverType::LinearSearch_LBFGS);

private:
    /**
     * @brief Function to find connected regions which peak above or below a given threshold.
     *
     * @param[in] data (arma::Mat): Image data.
     * @param[in] find_negative_sources (bool): Find also negative sources (with signal is -1)
     * @param[in] gaussian_fitting (bool): Compute auxiliary structures used for gaussian fitting.
     *
     * @return (uint) Number of valid labels
     */
    template <bool generateLabelMap>
    uint _label_detection_islands(const arma::Mat<real_t>& data, bool find_negative_sources = true, bool gaussian_fitting = true);
};
}

#endif /* SOURCE_FIND_H */
