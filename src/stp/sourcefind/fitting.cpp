/**
* @file fitting.cpp
* @brief Implementation of classes and functions of gaussian fitting.
*/

#include "fitting.h"

namespace stp {

double Gaussian2dFit::evaluate_point(const double x, const double y)
{
    const double& x_stddev = semimajor;
    const double& y_stddev = semiminor;

    // Auxiliary calculations for 2D gaussian function
    const double cost = cos(theta);
    const double cost2 = cost * cost;
    const double sint = sin(theta);
    const double sint2 = sint * sint;
    const double sin2t = sin(2.0 * theta);
    const double xstd2 = x_stddev * x_stddev;
    const double ystd2 = y_stddev * y_stddev;
    const double a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
    const double b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
    const double c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));
    const double xdiff = x - x_centre;
    const double ydiff = y - y_centre;

    return amplitude * exp(-(a * xdiff * xdiff
                           + b * xdiff * ydiff
                           + c * ydiff * ydiff));
}

bool GaussianAnalytic::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
    const double amplitude = parameters[0][0];
    const double x_centre = parameters[0][1];
    const double y_centre = parameters[0][2];
    const double x_stddev = parameters[0][3];
    const double y_stddev = parameters[0][4];
    const double theta = parameters[0][5];

    // Auxiliary calculations for 2D gaussian function
    const double cost = cos(theta);
    const double cost2 = cost * cost;
    const double sint = sin(theta);
    const double sint2 = sint * sint;
    const double sin2t = sin(2.0 * theta);
    const double xstd2 = x_stddev * x_stddev;
    const double ystd2 = y_stddev * y_stddev;
    const double a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
    const double b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
    const double c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));

    // Compute residual
    const double xdiff = _x - x_centre;
    const double ydiff = _y - y_centre;
    const double xdiff2 = xdiff * xdiff;
    const double ydiff2 = ydiff * ydiff;
    const double g = amplitude * exp(-((a * xdiff2) + (b * xdiff * ydiff) + (c * ydiff2)));
    residuals[0] = g - (double)_data;

    if (!jacobians)
        return true;
    double* jacobian = jacobians[0];
    if (!jacobian)
        return true;

    const double cos2t = cos(2.0 * theta);
    const double xstd3 = xstd2 * x_stddev;
    const double ystd3 = ystd2 * y_stddev;
    const double da_dtheta = (sint * cost * ((1.0 / ystd2) - (1.0 / xstd2)));
    const double da_dx_stddev = -cost2 / xstd3;
    const double da_dy_stddev = -sint2 / ystd3;
    const double db_dtheta = (cos2t / xstd2) - (cos2t / ystd2);
    const double db_dx_stddev = -sin2t / xstd3;
    const double db_dy_stddev = sin2t / ystd3;
    const double dc_dtheta = -da_dtheta;
    const double dc_dx_stddev = -sint2 / xstd3;
    const double dc_dy_stddev = -cost2 / ystd3;

    // Compute jacobian
    double dg_dA = g / amplitude;
    double dg_dx_mean = g * ((2.0 * a * xdiff) + (b * ydiff));
    double dg_dy_mean = g * ((b * xdiff) + (2.0 * c * ydiff));
    double dg_dx_stddev = g * (-(da_dx_stddev * xdiff2 + db_dx_stddev * xdiff * ydiff + dc_dx_stddev * ydiff2));
    double dg_dy_stddev = g * (-(da_dy_stddev * xdiff2 + db_dy_stddev * xdiff * ydiff + dc_dy_stddev * ydiff2));
    double dg_dtheta = g * (-(da_dtheta * xdiff2 + db_dtheta * xdiff * ydiff + dc_dtheta * ydiff2));

    jacobian[0] = dg_dA;
    jacobian[1] = dg_dx_mean;
    jacobian[2] = dg_dy_mean;
    jacobian[3] = dg_dx_stddev;
    jacobian[4] = dg_dy_stddev;
    jacobian[5] = dg_dtheta;

    return true;
}

bool GaussianAnalyticAllResiduals::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const
{
    // Gaussian parameters
    const double amplitude = parameters[0][0];
    const double x_centre = parameters[0][1];
    const double y_centre = parameters[0][2];
    const double x_stddev = parameters[0][3];
    const double y_stddev = parameters[0][4];
    const double theta = parameters[0][5];

    // Auxiliary calculations for 2D gaussian function
    const double cost = cos(theta);
    const double cost2 = cost * cost;
    const double sint = sin(theta);
    const double sint2 = sint * sint;
    const double sin2t = sin(2.0 * theta);
    const double xstd2 = x_stddev * x_stddev;
    const double ystd2 = y_stddev * y_stddev;
    const double a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2));
    const double b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2));
    const double c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2));
    double cos2t;
    double xstd3;
    double ystd3;
    double da_dtheta;
    double da_dx_stddev;
    double da_dy_stddev;
    double db_dtheta;
    double db_dx_stddev;
    double db_dy_stddev;
    double dc_dtheta;
    double dc_dx_stddev;
    double dc_dy_stddev;
    double* jacobian = nullptr;

    // Auxiliary calculations for computation of jacobian, if required
    if (jacobians) {
        cos2t = cos(2.0 * theta);
        xstd3 = xstd2 * x_stddev;
        ystd3 = ystd2 * y_stddev;
        da_dtheta = (sint * cost * ((1.0 / ystd2) - (1.0 / xstd2)));
        da_dx_stddev = -cost2 / xstd3;
        da_dy_stddev = -sint2 / ystd3;
        db_dtheta = (cos2t / xstd2) - (cos2t / ystd2);
        db_dx_stddev = -sin2t / xstd3;
        db_dy_stddev = sin2t / ystd3;
        dc_dtheta = -da_dtheta;
        dc_dx_stddev = -sint2 / xstd3;
        dc_dy_stddev = -cost2 / ystd3;
        jacobian = jacobians[0];
    }

#ifndef FFTSHIFT
    int h_shift = (int)(_data.n_cols / 2);
    int v_shift = (int)(_data.n_rows / 2);
#endif

    // Compute residuals on positions where label map is equal to label_idx
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

            const double xdiff = x - x_centre;
            const double ydiff = y - y_centre;
            const double xdiff2 = xdiff * xdiff;
            const double ydiff2 = ydiff * ydiff;
            const double g = amplitude * exp(-((a * xdiff2) + (b * xdiff * ydiff) + (c * ydiff2)));
            residuals[0] = g - (double)_data.at(jj, ii);
            residuals++;

            // Compute jacobian
            if (jacobians) {
                if (!jacobian)
                    continue;

                double dg_dA = g / amplitude;
                double dg_dx_mean = g * ((2.0 * a * xdiff) + (b * ydiff));
                double dg_dy_mean = g * ((b * xdiff) + (2.0 * c * ydiff));
                double dg_dx_stddev = g * (-(da_dx_stddev * xdiff2 + db_dx_stddev * xdiff * ydiff + dc_dx_stddev * ydiff2));
                double dg_dy_stddev = g * (-(da_dy_stddev * xdiff2 + db_dy_stddev * xdiff * ydiff + dc_dy_stddev * ydiff2));
                double dg_dtheta = g * (-(da_dtheta * xdiff2 + db_dtheta * xdiff * ydiff + dc_dtheta * ydiff2));

                jacobian[0] = dg_dA;
                jacobian[1] = dg_dx_mean;
                jacobian[2] = dg_dy_mean;
                jacobian[3] = dg_dx_stddev;
                jacobian[4] = dg_dy_stddev;
                jacobian[5] = dg_dtheta;
                jacobian += 6;
            }
        }
    }

    return true;
}
}
