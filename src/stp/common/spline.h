/*
 * spline.h
 *
 * simple cubic spline interpolation library
 *
 * ---------------------------------------------------------------------
 * Copyright (C) 2018 Luis Lucas (luisfrlucas at gmail.com)
 * Copyright (C) 2011, 2014 Tino Kluge (ttk448 at gmail.com)
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  as published by the Free Software Foundation; either version 2
 *  of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * ---------------------------------------------------------------------
 *
 */

#ifndef TK_SPLINE_H
#define TK_SPLINE_H

#include "../types.h"
#include <armadillo>
#include <assert.h>

namespace tk {

// band matrix solver
class band_matrix {
private:
    arma::Mat<real_t> m_upper; // upper band
    arma::Mat<real_t> m_lower; // lower band
public:
    band_matrix() {} // constructor
    band_matrix(int dim, int n_u, int n_l); // constructor
    ~band_matrix() {} // destructor
    void setsize(int dim, int n_u, int n_l); // init with dim,n_u,n_l
    int dim() const; // matrix dimension
    int num_upper() const
    {
        return m_upper.n_cols - 1;
    }
    int num_lower() const
    {
        return m_lower.n_cols - 1;
    }
    // access operator
    real_t& operator()(int i, int j); // write
    real_t operator()(int i, int j) const; // read
    // we can store an additional diogonal (in m_lower)
    real_t& saved_diag(int i);
    real_t saved_diag(int i) const;
    void lu_decompose();
    arma::Col<real_t> r_solve(const arma::Col<real_t>& b) const;
    arma::Col<real_t> l_solve(const arma::Col<real_t>& b) const;
    arma::Col<real_t> lu_solve(const arma::Col<real_t>& b, bool is_lu_decomposed = false);
};

/**
 * @brief The spline class
 *
 * Performs cubic or linear spline interpolation
 */
template <bool isCubic = true>
class spline {
public:
    enum bd_type {
        first_deriv = 1,
        second_deriv = 2
    };

private:
    arma::Col<real_t> m_x, m_y; // x,y coordinates of points
    // interpolation parameters
    // f(x) = a*(x-x_i)^3 + b*(x-x_i)^2 + c*(x-x_i) + y_i
    arma::Col<real_t> m_a, m_b, m_c; // spline coefficients
    real_t m_b0, m_c0; // for left extrapol
    bd_type m_left, m_right;
    real_t m_left_value, m_right_value;
    bool m_force_linear_extrapolation;

public:
    /**
     * @brief Spline class constructor
     *
     * Set default boundary condition to be zero curvature at both ends
     */
    spline()
        : m_left(second_deriv)
        , m_right(second_deriv)
        , m_left_value(0.0)
        , m_right_value(0.0)
        , m_force_linear_extrapolation(false)
    {
    }

    /**
     * @brief Set default boundary conditions
     * Optional, but if called it has to come be before set_points()
     */
    void set_boundary(bd_type left, real_t left_value,
        bd_type right, real_t right_value,
        bool force_linear_extrapolation = false)
    {
        assert(m_x.size() == 0); // set_points() must not have happened yet
        m_left = left;
        m_right = right;
        m_left_value = left_value;
        m_right_value = right_value;
        m_force_linear_extrapolation = force_linear_extrapolation;
    }

    /**
     * @brief Set points for interpolation
     * If copy parameter is false, data is not stored internally.
     * This provides computational savings since memcopy operations are not performed.
     *
     * @param[in] x (arma::Col): Input vector with point locations.
     * @param[in] y (arma::Col): Input vector with the corresponding function values.
     * @param[in] copy (bool): If true, copy the input arrays x and y to internal structures.
     */
    void set_points(const arma::Col<real_t>& x, const arma::Col<real_t>& y, bool copy = true)
    {
        assert(x.size() == y.size());
        assert(x.size() > 2);
        if (copy) {
            m_x = x;
            m_y = y;
        }
        int n = x.size();
        // TODO: maybe sort x and y, rather than returning an error
        for (int i = 0; i < n - 1; i++) {
            assert(x[i] < x[i + 1]);
        }

        if (isCubic == true) { // cubic spline interpolation
            // setting up the matrix and right hand side of the equation system
            // for the parameters b[]
            band_matrix A(n, 1, 1);
            arma::Col<real_t> rhs(n);
            for (int i = 1; i < n - 1; i++) {
                A(i, i - 1) = 1.0 / 3.0 * (x[i] - x[i - 1]);
                A(i, i) = 2.0 / 3.0 * (x[i + 1] - x[i - 1]);
                A(i, i + 1) = 1.0 / 3.0 * (x[i + 1] - x[i]);
                rhs[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
            }
            // boundary conditions
            if (m_left == spline::second_deriv) {
                // 2*b[0] = f''
                A(0, 0) = 2.0;
                A(0, 1) = 0.0;
                rhs[0] = m_left_value;
            } else if (m_left == spline::first_deriv) {
                // c[0] = f', needs to be re-expressed in terms of b:
                // (2b[0]+b[1])(x[1]-x[0]) = 3 ((y[1]-y[0])/(x[1]-x[0]) - f')
                A(0, 0) = 2.0 * (x[1] - x[0]);
                A(0, 1) = 1.0 * (x[1] - x[0]);
                rhs[0] = 3.0 * ((y[1] - y[0]) / (x[1] - x[0]) - m_left_value);
            } else {
                assert(false);
            }
            if (m_right == spline::second_deriv) {
                // 2*b[n-1] = f''
                A(n - 1, n - 1) = 2.0;
                A(n - 1, n - 2) = 0.0;
                rhs[n - 1] = m_right_value;
            } else if (m_right == spline::first_deriv) {
                // c[n-1] = f', needs to be re-expressed in terms of b:
                // (b[n-2]+2b[n-1])(x[n-1]-x[n-2])
                // = 3 (f' - (y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
                A(n - 1, n - 1) = 2.0 * (x[n - 1] - x[n - 2]);
                A(n - 1, n - 2) = 1.0 * (x[n - 1] - x[n - 2]);
                rhs[n - 1] = 3.0 * (m_right_value - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
            } else {
                assert(false);
            }

            // solve the equation system to obtain the parameters b[]
            m_b = A.lu_solve(rhs);

            // calculate parameters a[] and c[] based on b[]
            m_a.set_size(n);
            m_c.set_size(n);
            for (int i = 0; i < n - 1; i++) {
                m_a[i] = 1.0 / 3.0 * (m_b[i + 1] - m_b[i]) / (x[i + 1] - x[i]);
                m_c[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                    - 1.0 / 3.0 * (2.0 * m_b[i] + m_b[i + 1]) * (x[i + 1] - x[i]);
            }
        } else { // linear interpolation
            m_c.set_size(n);
            for (int i = 0; i < n - 1; i++) {
                m_c[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
            }
        }
        if (isCubic) {
            // for left extrapolation coefficients
            m_b0 = (m_force_linear_extrapolation == false) ? m_b[0] : 0.0;
            m_c0 = m_c[0];

            // for the right extrapolation coefficients
            // f_{n-1}(x) = b*(x-x_{n-1})^2 + c*(x-x_{n-1}) + y_{n-1}
            real_t h = x[n - 1] - x[n - 2];
            // m_b[n-1] is determined by the boundary condition
            m_a[n - 1] = 0.0;
            m_c[n - 1] = 3.0 * m_a[n - 2] * h * h + 2.0 * m_b[n - 2] * h + m_c[n - 2]; // = f'_{n-2}(x_{n-1})
            if (m_force_linear_extrapolation == true)
                m_b[n - 1] = 0.0;
        } else {
            m_c0 = m_c[0];
            m_c[n - 1] = m_c[n - 2];
        }
    }

    // Find the lower bound index
    inline int mx_lower_bound(int first, int last, real_t x) const
    {
        while (first < last) {
            int mid = (first + last) >> 1;
            if (x > m_x[mid]) {
                first = mid + 1;
            } else {
                last = mid;
            }
        }
        first = std::max(first - 1, 0);
        return first;
    }

    /**
     * @brief Compute interpolation value at indicated location.
     *
     * This function requires internal copies of the point location and function arrays.
     * Copies are performed when calling set_points() function with copy=true.
     *
     * @param[in] x (real_t): Interpolation location.
     */
    real_t operator()(real_t x) const
    {
        size_t n = m_x.size();
        // find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
        int idx = mx_lower_bound(0, m_x.n_elem, x);

        real_t h = x - m_x[idx];
        real_t interpol;
        if (x < m_x[0]) {
            // extrapolation to the left
            if (isCubic) {
                interpol = (m_b0 * h + m_c0) * h + m_y[0];
            } else {
                interpol = m_c0 * h + m_y[0];
            }
        } else if (x > m_x[n - 1]) {
            // extrapolation to the right
            if (isCubic) {
                interpol = (m_b[n - 1] * h + m_c[n - 1]) * h + m_y[n - 1];
            } else {
                interpol = m_c[n - 1] * h + m_y[n - 1];
            }
        } else {
            // interpolation
            if (isCubic) {
                interpol = ((m_a[idx] * h + m_b[idx]) * h + m_c[idx]) * h + m_y[idx];
            } else {
                interpol = m_c[idx] * h + m_y[idx];
            }
        }
        return interpol;
    }

    /**
     * @brief Compute interpolation value at indicated location.
     *
     * This function does not use internal copies of the point location and function arrays.
     * Must be used when set_points() function is called when with copy=false.
     * This is so for better execution performance.
     *
     * @param[in] x (real_t): Interpolation location.
     * @param[in] idx (int): Index of the lower bound location.
     * @param[in] dm_x (real_t): Lower bound location value.
     * @param[in] dm_y (real_t): Lower bound function value.
     */
    real_t operator()(real_t& x, int idx, const real_t& dm_x, const real_t& dm_y) const
    {
        real_t h = x - dm_x;
        if (isCubic)
            return ((m_a[idx] * h + m_b[idx]) * h + m_c[idx]) * h + dm_y;
        else
            return m_c[idx] * h + dm_y;
    }

    // Compute derivatives
    real_t deriv(int order, real_t x) const
    {
        assert(order > 0);

        size_t n = m_x.size();
        // find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
        int idx = mx_lower_bound(0, m_x.n_elem, x);

        real_t h = x - m_x[idx];
        real_t interpol;
        if (x < m_x[0]) {
            // extrapolation to the left
            switch (order) {
            case 1:
                interpol = 2.0 * m_b0 * h + m_c0;
                break;
            case 2:
                interpol = 2.0 * m_b0 * h;
                break;
            default:
                interpol = 0.0;
                break;
            }
        } else if (x > m_x[n - 1]) {
            // extrapolation to the right
            switch (order) {
            case 1:
                interpol = 2.0 * m_b[n - 1] * h + m_c[n - 1];
                break;
            case 2:
                interpol = 2.0 * m_b[n - 1];
                break;
            default:
                interpol = 0.0;
                break;
            }
        } else {
            // interpolation
            switch (order) {
            case 1:
                interpol = (3.0 * m_a[idx] * h + 2.0 * m_b[idx]) * h + m_c[idx];
                break;
            case 2:
                interpol = 6.0 * m_a[idx] * h + 2.0 * m_b[idx];
                break;
            case 3:
                interpol = 6.0 * m_a[idx];
                break;
            default:
                interpol = 0.0;
                break;
            }
        }
        return interpol;
    }
};

/**
 * @brief The linear spline class
 *
 * Performs linear spline interpolation
 */
class linearspline {
private:
    arma::Col<real_t> m_x, m_y; // x,y coordinates of points
    // interpolation parameters
    // f(x) = a*(x-x_i) + y_i
    arma::Col<real_t> m_a; // spline coefficients

public:
    /**
     * @brief Linear spline constructor
     */
    linearspline()
    {
    }

    void set_points(const arma::Col<real_t>& x, const arma::Col<real_t>& y, bool copy = true);
    real_t operator()(real_t x) const;
    real_t operator()(real_t& x, int idx, const real_t& dm_x, const real_t& dm_y) const;
    int mx_lower_bound(int first, int last, real_t x) const;
};

} // namespace tk

#endif /* TK_SPLINE_H */
