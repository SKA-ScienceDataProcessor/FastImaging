/*
 * spline.cpp
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

#include "spline.h"
#include <algorithm>
#include <cassert>
#include <cstdio>

namespace tk {

// band_matrix implementation

band_matrix::band_matrix(int dim, int n_u, int n_l)
{
    setsize(dim, n_u, n_l);
}
void band_matrix::setsize(int dim, int n_u, int n_l)
{
    assert(dim > 0);
    assert(n_u >= 0);
    assert(n_l >= 0);
    m_upper.set_size(dim, n_u + 1);
    m_lower.set_size(dim, n_l + 1);
}
int band_matrix::dim() const
{
    if (m_upper.n_cols > 0) {
        return m_upper.n_rows;
    } else {
        return 0;
    }
}

// defines the new operator (), so that we can access the elements
// by A(i,j), index going from i=0,...,dim()-1
real_t& band_matrix::operator()(int i, int j)
{
    int k = j - i; // what band is the entry
    assert((i >= 0) && (i < dim()) && (j >= 0) && (j < dim()));
    assert((-num_lower() <= k) && (k <= num_upper()));
    // k=0 -> diogonal, k<0 lower left part, k>0 upper right part
    if (k >= 0)
        return m_upper(i, k);
    else
        return m_lower(i, -k);
}
real_t band_matrix::operator()(int i, int j) const
{
    int k = j - i; // what band is the entry
    assert((i >= 0) && (i < dim()) && (j >= 0) && (j < dim()));
    assert((-num_lower() <= k) && (k <= num_upper()));
    // k=0 -> diogonal, k<0 lower left part, k>0 upper right part
    if (k >= 0)
        return m_upper(i, k);
    else
        return m_lower(i, -k);
}
// second diag (used in LU decomposition), saved in m_lower
real_t band_matrix::saved_diag(int i) const
{
    assert((i >= 0) && (i < dim()));
    return m_lower(i, 0);
}
real_t& band_matrix::saved_diag(int i)
{
    assert((i >= 0) && (i < dim()));
    return m_lower(i, 0);
}

// LR-Decomposition of a band matrix
void band_matrix::lu_decompose()
{
    int i_max, j_max;
    int j_min;
    real_t x;

    // preconditioning
    // normalize column i so that a_ii=1
    for (int i = 0; i < this->dim(); i++) {
        assert(this->operator()(i, i) != 0.0);
        this->saved_diag(i) = 1.0 / this->operator()(i, i);
        j_min = std::max(0, i - this->num_lower());
        j_max = std::min(this->dim() - 1, i + this->num_upper());
        for (int j = j_min; j <= j_max; j++) {
            this->operator()(i, j) *= this->saved_diag(i);
        }
        this->operator()(i, i) = 1.0; // prevents rounding errors
    }

    // Gauss LR-Decomposition
    for (int k = 0; k < this->dim(); k++) {
        i_max = std::min(this->dim() - 1, k + this->num_lower()); // num_lower not a mistake!
        for (int i = k + 1; i <= i_max; i++) {
            assert(this->operator()(k, k) != 0.0);
            x = -this->operator()(i, k) / this->operator()(k, k);
            this->operator()(i, k) = -x; // assembly part of L
            j_max = std::min(this->dim() - 1, k + this->num_upper());
            for (int j = k + 1; j <= j_max; j++) {
                // assembly part of R
                this->operator()(i, j) = this->operator()(i, j) + x * this->operator()(k, j);
            }
        }
    }
}
// solves Ly=b
arma::Col<real_t> band_matrix::l_solve(const arma::Col<real_t>& b) const
{
    assert(this->dim() == (int)b.size());
    arma::Col<real_t> x(this->dim());
    int j_start;
    real_t sum;
    for (int i = 0; i < this->dim(); i++) {
        sum = 0;
        j_start = std::max(0, i - this->num_lower());
        for (int j = j_start; j < i; j++)
            sum += this->operator()(i, j) * x[j];
        x[i] = (b[i] * this->saved_diag(i)) - sum;
    }
    return x;
}
// solves Rx=y
arma::Col<real_t> band_matrix::r_solve(const arma::Col<real_t>& b) const
{
    assert(this->dim() == (int)b.size());
    arma::Col<real_t> x(this->dim());
    int j_stop;
    real_t sum;
    for (int i = this->dim() - 1; i >= 0; i--) {
        sum = 0;
        j_stop = std::min(this->dim() - 1, i + this->num_upper());
        for (int j = i + 1; j <= j_stop; j++)
            sum += this->operator()(i, j) * x[j];
        x[i] = (b[i] - sum) / this->operator()(i, i);
    }
    return x;
}

arma::Col<real_t> band_matrix::lu_solve(const arma::Col<real_t>& b,
    bool is_lu_decomposed)
{
    assert(this->dim() == (int)b.size());
    arma::Col<real_t> x, y;
    if (is_lu_decomposed == false) {
        this->lu_decompose();
    }
    y = this->l_solve(b);
    x = this->r_solve(y);
    return x;
}

// linear spline implementation
// -----------------------
void linearspline::set_points(const arma::Col<real_t>& x, const arma::Col<real_t>& y, bool copy)
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

    m_a.set_size(n);
    for (int i = 0; i < n - 1; i++) {
        m_a[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    }
    m_a[n - 1] = m_a[n - 2];
}

inline int linearspline::mx_lower_bound(int first, int last, real_t x) const
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

real_t linearspline::operator()(real_t x) const
{
    size_t n = m_x.size();
    // find the closest point m_x[idx] < x, idx=0 even if x<m_x[0]
    int idx = mx_lower_bound(0, m_x.n_elem, x);

    real_t h = x - m_x[idx];
    real_t interpol;
    if (x < m_x[0]) {
        // extrapolation to the left
        interpol = m_a[0] * h + m_y[0];
    } else if (x > m_x[n - 1]) {
        // extrapolation to the right
        interpol = m_a[n - 1] * h + m_y[n - 1];
    } else {
        // interpolation
        interpol = m_a[idx] * h + m_y[idx];
    }
    return interpol;
}

real_t linearspline::operator()(real_t& x, int idx, const real_t& dm_x, const real_t& dm_y) const
{
    real_t h = x - dm_x;
    return m_a[idx] * h + dm_y;
}

} // namespace tk
