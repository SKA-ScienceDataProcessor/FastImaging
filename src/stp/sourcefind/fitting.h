/**
* @file fitting.h
* @brief Classes and function prototypes of gaussian fitting.
*/

#ifndef FITTING_H
#define FITTING_H

#include "../types.h"
#include <armadillo>
#include <ceres/ceres.h>

namespace stp {

/**
 * @brief Represents bounding box positions: top, bottom, left and right margins
 */
struct BoundingBox {
    int top;
    int bottom;
    int left;
    int right;

    /**
     * @brief Default BoundingBox constructor.
     *
     *  All parameters set to zero.
     */
    BoundingBox()
        : top(0)
        , bottom(0)
        , left(0)
        , right(0)
    {
    }

    /**
     * @brief BoundingBox constructor that sets all parameters.
     *
     * @param[in] in_top (double): Bounding box top row.
     * @param[in] in_bottom (double): Bounding box bottom row.
     * @param[in] in_left (double): Bounding box left column.
     * @param[in] in_right (double): Bounding box right column.
     */
    BoundingBox(int in_top, int in_bottom, int in_left, int in_right)
        : top(in_top)
        , bottom(in_bottom)
        , left(in_left)
        , right(in_right)
    {
    }

    /**
     * @brief Bounding box width.
     *
     * @return (double) width.
     */
    int get_width()
    {
        return right - left + 1;
    }

    /**
     * @brief Bounding box height.
     *
     * @return (double) height.
     */
    int get_height()
    {
        return bottom - top + 1;
    }
};

/**
 * @brief The Gaussian2dFit struct
 *
 *   Data structure for representing Gaussian Fits.
 *   It refers to semimajor/semiminor axis length rather than x_std_dev / y_std_dev.
 *   Always assumes that `x_std_dev > y_std_dev`, or equivalently, that theta describes the
 *   rotation in the counterclockwise sense of the semimajor axis from the positive x-direction.
 *   For fits returned where this does not happen to be true, we swap semi-major/minor and adjust theta accordingly.
 *   All values are in units of pixels, except for theta which has units of radians.
 */
struct Gaussian2dFit {
    double amplitude;
    double x_centre;
    double y_centre;
    double semimajor;
    double semiminor;
    double theta;

    /**
     * @brief Default Gaussian2dFit constructor.
     *
     * All parameters set to zero.
     */
    Gaussian2dFit()
        : amplitude(0.0)
        , x_centre(0.0)
        , y_centre(0.0)
        , semimajor(0.0)
        , semiminor(0.0)
        , theta(0.0)
    {
    }

    /**
     * @brief Gaussian2dFit constructor that sets all parameters.
     *
     * @param[in] in_amplitude (double): Amplitude of the Gaussian.
     * @param[in] in_x_centre (double): Mean of the Gaussian in x.
     * @param[in] in_y_centre (double): Mean of the Gaussian in y.
     * @param[in] in_semimajor (double): Semimajor axis length of the Gaussian. Corresponds to standard deviation of the Gaussian in x before rotating by theta.
     * @param[in] in_semiminor (double): Semiminor axis length of the Gaussian. Corresponds to standard deviation of the Gaussian in y before rotating by theta.
     * @param[in] in_theta (double): Rotation angle in radians. The rotation angle increases counterclockwise.
     */
    Gaussian2dFit(double in_amplitude, double in_x_centre, double in_y_centre, double in_semimajor, double in_semiminor, double in_theta)
        : amplitude(in_amplitude)
        , x_centre(in_x_centre)
        , y_centre(in_y_centre)
        , semimajor(in_semimajor)
        , semiminor(in_semiminor)
        , theta(in_theta)
    {
    }

    /**
     * @brief Evaluate 2D gaussian model at the given x,y point.
     *
     * @param[in] x (double): Point coordinate x.
     * @param[in] y (double): Point coordinate y.
     *
     * @return (double) 2D gaussian value at point x,y.
     */
    double evaluate_point(const double x, const double y);
};

/**
 * @brief Functor that computes the residual on the given point x,y between source data and 2D gaussian model.
 *
 * Computes residual on the given point x,y between the source data and the 2D gaussian model.
 * Required by the ceres solver when performing non-linear least-squares optimisation based on automatic differentiation method.
 */
class GaussianResidual {
public:
    /**
     * @brief GaussianResidual constructor.
     *
     * @param[in] data (double): Data value at given point x,y.
     * @param[in] x (double): Point coordinate x.
     * @param[in] y (double): Point coordinate y.
     */
    GaussianResidual(const double data, const double x, const double y)
        : _data(data)
        , _x(x)
        , _y(y)
    {
    }

    /**
     * @brief Operator that computes residuals of between data and 2D gaussian model.
     *
     * @param[in] params (T*): Guassian function parameters to be optimised.
     * @param[in] residual (T*): Residual values.
     */
    template <typename T>
    bool operator()(const T* const params, T* residual) const
    {
        const T& amplitude = params[0];
        const T& x_centre = params[1];
        const T& y_centre = params[2];
        const T& x_stddev = params[3];
        const T& y_stddev = params[4];
        const T& theta = params[5];

        // Auxiliary calculations for 2D gaussian function
        const T cost = cos(theta);
        const T cost2 = cost * cost;
        const T sint = sin(theta);
        const T sint2 = sint * sint;
        const T sin2t = sin(2.0 * theta);
        const T xstd2 = x_stddev * x_stddev;
        const T ystd2 = y_stddev * y_stddev;
        const T a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
        const T b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
        const T c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));
        const T xdiff = _x - x_centre;
        const T ydiff = _y - y_centre;

        // Compute residual
        residual[0] = amplitude * exp(-(a * xdiff * xdiff
                                      + b * xdiff * ydiff
                                      + c * ydiff * ydiff))
            - _data;

        return true;
    }

private:
    const double _data;
    const double _x;
    const double _y;
};

/**
 * @brief Functor that computes all residuals between source data and 2D gaussian model.
 *
 * Computes all the residuals between the source data and the 2D gaussian model.
 * Required by the ceres solver when performing non-linear least-squares optimisation based on automatic differentiation method.
 */
class GaussianAllResiduals {
public:
    /**
     * @brief GaussianAllResiduals constructor
     *
     * @param[in] data (arma::Mat<real_t>): Image matrix.
     * @param[in] label_map (arma::Mat<int>): Label map matrix.
     * @param[in] label_idx (int): Label index.
     * @param[in] bbox (BoundingBox): Bounding box defined around the source.
     */
    GaussianAllResiduals(const arma::Mat<real_t>& data, const arma::Mat<int>& label_map, const BoundingBox& box, const int label_idx)
        : _data(data)
        , _label_map(label_map)
        , _label_idx(label_idx)
        , _box(box)
    {
    }

    /**
     * @brief Operator that computes residuals of between data and 2D gaussian model.
     *
     * @param[in] params (T*): Guassian function parameters to be optimised.
     * @param[in] residual (T*): Residual values.
     */
    template <typename T>
    bool operator()(const T* const params, T* residual) const
    {
        const T& amplitude = params[0];
        const T& x_centre = params[1];
        const T& y_centre = params[2];
        const T& x_stddev = params[3];
        const T& y_stddev = params[4];
        const T& theta = params[5];

        // Auxiliary calculations for 2D gaussian function
        const T cost = cos(theta);
        const T cost2 = cost * cost;
        const T sint = sin(theta);
        const T sint2 = sint * sint;
        const T sin2t = sin(2.0 * theta);
        const T xstd2 = x_stddev * x_stddev;
        const T ystd2 = y_stddev * y_stddev;
        const T a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
        const T b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
        const T c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));

#ifndef FFTSHIFT
        int h_shift = (int)(_data.n_cols / 2);
        int v_shift = (int)(_data.n_rows / 2);
#endif

        // Compute residuals on positions where label map corresponds to label_idx
        for (int i = _box.left; i <= _box.right; ++i) {
            for (int j = _box.top; j <= _box.bottom; ++j) {
                const double x = (double)(i);
                const double y = (double)(j);
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

                const T xdiff = x - x_centre;
                const T ydiff = y - y_centre;

                residual[0] = amplitude * exp(-(a * xdiff * xdiff
                                              + b * xdiff * ydiff
                                              + c * ydiff * ydiff))
                    - (double)_data.at(jj, ii);

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
 * @brief Computes residual and jacobian on the given point x,y for gaussian fitting using analytic derivatives.
 *
 * Class derived from ceres::SizedCostFunction that computes function cost and jacobian on the given point x,y.
 * Required by the ceres solver when performing non-linear least-squares optimisation based on analytic derivatives.
 */
class GaussianAnalytic : public ceres::SizedCostFunction<1, 6> {
public:
    /**
     * @brief GaussianAnalytic constructor.
     *
     * @param[in] data (double): Data value at given point x,y.
     * @param[in] x (double): Point coordinate x.
     * @param[in] y (double): Point coordinate y.
     */
    GaussianAnalytic(const double data, const double x, const double y)
        : _data(data)
        , _x(x)
        , _y(y)
    {
    }

    /**
     * @brief Compute the residual and the Jacobian matrix.
     *
     *  Virtual method from ceres::SizedCostFunction.
     */
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

private:
    const double _data;
    const double _x;
    const double _y;
};

/**
 * @brief Computes all source residuals and jacobians for gaussian fitting using analytic derivatives.
 *
 * Class derived from ceres::SizedCostFunction that computes all residuals and jacobians of the source.
 * Required by the ceres solver when performing non-linear least-squares optimisation based on analytic derivatives.
 */
class GaussianAnalyticAllResiduals : public ceres::CostFunction {
public:
    /**
         * @brief GaussianAnalyticAllResiduals constructor.
         *
         * @param[in] data (arma::Mat<real_t>): Image matrix.
         * @param[in] label_map (arma::Mat<int>): Label map matrix.
         * @param[in] bbox (BoundingBox): Bounding box defined around the source.
         * @param[in] label_idx (int): Label index.
         * @param[in] num_residuals (int): Number of residuals.
         * @param[in] parameter_block_size (int): Size of parameter block.
         */
    GaussianAnalyticAllResiduals(const arma::Mat<real_t>& data, const arma::Mat<int>& label_map, const BoundingBox& box,
        const int label_idx, const int num_residuals, const int parameter_block_size)
        : _data(data)
        , _label_map(label_map)
        , _label_idx(label_idx)
        , _box(box)
    {
        // Set number of residuals
        assert(num_residuals > 0);
        set_num_residuals(num_residuals);
        // Set size of parameter block
        assert(parameter_block_size > 0);
        mutable_parameter_block_sizes()->push_back(parameter_block_size);
    }

    /**
     * @brief Compute the residual vector and the Jacobian matrices.
     *
     *  Virtual method from ceres::SizedCostFunction.
     */
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

private:
    const arma::Mat<real_t>& _data;
    const arma::Mat<int>& _label_map;
    const int _label_idx;
    const BoundingBox _box;
};
}

#endif /* FITTING_H */
