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
#include <ceres/ceres.h>
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
 * @brief Represents bounding box positions: top, bottom, left and right margins
 */
struct BoundingBox {
    int top;
    int bottom;
    int left;
    int right;

    BoundingBox()
        : top(0)
        , bottom(0)
        , left(0)
        , right(0)
    {
    }

    BoundingBox(int in_top, int in_bottom, int in_left, int in_right)
        : top(in_top)
        , bottom(in_bottom)
        , left(in_left)
        , right(in_right)
    {
    }
};

/**
 * @brief Functor for the ceres solver.
 *
 * Functor that computes the residuals of gaussian model fitting. Used by ceres solver library.
 */
class GaussianResiduals {
public:
    /**
     * @brief Constructor
     *
     * @param[in] data (arma::Mat<real_t>): Image matrix.
     * @param[in] label_map (arma::Mat<int>): Label map matrix.
     * @param[in] bbox (BoundingBox): Bounding box defined around the source.
     * @param[in] label_idx (int): Label index.
     */
    GaussianResiduals(const arma::Mat<real_t>& data, const arma::Mat<int>& label_map, const BoundingBox& box, const int label_idx)
        : _data(data)
        , _label_map(label_map)
        , _label_idx(label_idx)
        , _box(box)
    {
    }

    /**
     * @brief Operator that computes residuals of gaussian model fitting.
     *
     * @param[in] params (T*): Guassian function parameters to be optimised.
     * @param[in] residual (T*): Residual values.
     */
    template <typename T>
    bool operator()(const T* const params, T* residual) const
    {
        const T& amplitude = params[0];
        const T& x0 = params[1];
        const T& y0 = params[2];
        const T& x_stddev = params[3];
        const T& y_stddev = params[4];
        const T& theta = params[5];

        // Upper and/or lower bounds constraints on the parameters
        // x0,y0 must be located within the bounding box
        if ((x0 < (double)_box.left) || (x0 > (double)_box.right))
            return false;
        if ((y0 < (double)_box.top) || (y0 > (double)_box.bottom))
            return false;
        // x,y stddev must be positive
        if (x_stddev <= 0.0)
            return false;
        if (y_stddev <= 0.0)
            return false;
        // Theta varies between 0 and 2*pi
        if (theta < 0.0)
            return false;
        if (theta > (2 * arma::datum::pi))
            return false;

        // Auxiliary calculations for gaussian function
        T a = (cos(theta) * cos(theta) / (2.0 * x_stddev * x_stddev)) + (sin(theta) * sin(theta) / (2.0 * y_stddev * y_stddev));
        T b = (sin(2.0 * theta) / (2.0 * x_stddev * x_stddev)) - (sin(2.0 * theta) / (2.0 * y_stddev * y_stddev));
        T c = (sin(theta) * sin(theta) / (2.0 * x_stddev * x_stddev)) + (cos(theta) * cos(theta) / (2.0 * y_stddev * y_stddev));
#ifndef FFTSHIFT
        int h_shift = (int)(_data.n_cols / 2);
        int v_shift = (int)(_data.n_rows / 2);
#endif

        // Compute residuals on positions where label map is equal to label_idx
        for (int i = _box.left; i <= _box.right; ++i) {
            for (int j = _box.top; j <= _box.bottom; ++j) {
                const double col = (double)(i);
                const double row = (double)(j);
#ifdef FFTSHIFT
                const int& ii = i;
                const int& jj = j;
#else
                const int ii = i < h_shift ? i + h_shift : i - h_shift;
                const int jj = j < v_shift ? j + v_shift : j - v_shift;
#endif
                if (_label_map.at(jj, ii) != _label_idx) {
                    continue;
                }
                residual[0] = (double)_data.at(jj, ii) - (amplitude * exp(-a * (col - x0) * (col - x0) - 2.0 * b * (col - x0) * (row - y0) - c * (row - y0) * (row - y0)));
                residual++;
            }
        }
        return true;
    }

private:
    const arma::Mat<real_t>& _data;
    const arma::Mat<int>& _label_map;
    const int _label_idx;
    const BoundingBox _box;
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
    BoundingBox bounding_box;
    int l_box_width;
    int l_box_height;
    double g_amplitude;
    double g_x0;
    double g_y0;
    double g_x_stddev;
    double g_y_stddev;
    double g_theta;
    bool used_ceres;
    ceres::Solver::Summary summary;

    /**
     * @brief island_params default constructor
     */
    island_params() = default;

    /**
     * @brief island_params constructor
     *
     * Initialized with label index, and peak-pixel value, peak-pixel position and barycentre.
     *
     * @param[in] label (int): Index of region in label-map of source image
     * @param[in] l_extremum (real_t): the extremum value
     * @param[in] l_extremum_coord_y (int): the y-coordinate index of the extremum value
     * @param[in] l_extremum_coord_x (int): the x-coordinate index of the extremum value
     * @param[in] barycentre_y (real_t): the y-value of barycentric centre
     * @param[in] barycentre_x (real_t): the x-value of barycentric centre
     * @param[in] bbox (BoundingBox): the bounding box defined around the source
     */
    island_params(const int label, const real_t l_extremum, const int l_extremum_coord_y,
        const int l_extremum_coord_x, const real_t barycentre_y, const real_t barycentre_x,
        const BoundingBox& box = BoundingBox());

    /**
     * @brief Performs gaussian fitting
     *
     * Performs gaussian fitting using the bounding box information, image data matrix and label map matrix.
     *
     * @param[in] data (arma::Mat<real_t>): Image matrix
     * @param[in] label_map (arma::Mat<int>): Label map matrix
     */
    void fit_gaussian(const arma::Mat<real_t>& data, const arma::Mat<int>& label_map);

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
 * This can be useful for verifying / debugging the sourcefinder results,
 * and intermediate results can also be reused to save recalculation.
 *
 */
class source_find_image {

public:
    stp::MatStp<int> label_map;
    arma::Col<real_t> label_extrema_val_pos;
    arma::Col<real_t> label_extrema_val_neg;
    arma::uvec label_extrema_linear_idx_pos;
    arma::uvec label_extrema_linear_idx_neg;
    arma::ivec label_extrema_id_pos;
    arma::ivec label_extrema_id_neg;
    arma::mat label_extrema_barycentre_pos;
    arma::mat label_extrema_barycentre_neg;
    tbb::concurrent_vector<BoundingBox> label_extrema_boundingbox_pos;
    tbb::concurrent_vector<BoundingBox> label_extrema_boundingbox_neg;
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
     * @param[in] compute_barycentre (bool): Compute barycentric centre of each island.
     * @param[in] gaussian_fitting (bool): Perform gaussian fitting for each island.
     * @param[in] generate_labelmap (bool): Update the final label map by removing the sources below the detection threshold.
     */
    source_find_image(
        const arma::Mat<real_t>& input_data,
        double input_detection_n_sigma,
        double input_analysis_n_sigma,
        double input_rms_est = 0.0,
        bool find_negative_sources = true,
        uint sigma_clip_iters = 5,
        bool binapprox_median = false,
        bool compute_barycentre = true,
        bool gaussian_fitting = false,
        bool generate_labelmap = true);

private:
    /**
     * @brief Function to find connected regions which peak above or below a given threshold.
     *
     * @param[in] data (arma::Mat): Image data.
     * @param[in] find_negative_sources (bool): Find also negative sources (with signal is -1)
     * @param[in] compute_barycentre (bool): Compute barycentric centre of each island.
     * @param[in] gaussian_fitting (bool): Compute auxiliary structures used for gaussian fitting.
     *
     * @return (uint) Number of valid labels
     */
    template <bool generateLabelMap>
    uint _label_detection_islands(const arma::Mat<real_t>& data, bool find_negative_sources = true, bool compute_barycentre = true, bool gaussian_fitting = false);
};
}

#endif /* SOURCE_FIND_H */
