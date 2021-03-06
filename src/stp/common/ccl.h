/**
* @file ccl.h
* @brief Function prototypes of the connected component labeling.
*/

#ifndef CCL_H
#define CCL_H

#include "../global_macros.h"
#include "../types.h"
#include "matstp.h"
#include <armadillo>
#include <cassert>
#include <tbb/tbb.h>

namespace stp {

// Find the root of the tree of node i.
inline static uint
find_root(const uint* P, uint i)
{
    uint root = i;
    while (P[root] < root) {
        root = P[root];
    }
    return root;
}

// Make all nodes in the path of node i point to root.
inline static void set_root(uint* P, uint i, uint root)
{
    while (P[i] < i) {
        uint j = P[i];
        P[i] = root;
        i = j;
    }
    P[i] = root;
}

// Unite the two trees containing nodes i and j and return the new root.
inline static uint set_union(uint* P, uint i, uint j)
{
    uint root = find_root(P, i);
    if (i != j) {
        uint rootj = find_root(P, j);
        if (root > rootj) {
            root = rootj;
        }
        set_root(P, j, root);
    }
    set_root(P, i, root);
    return root;
}

/**
 * @brief LabelDataThread struct is used to store start column and detected label information of a thread
 */
struct LabelDataThread {

    /**
     * @brief LabelDataThread default constructor
     */
    LabelDataThread()
        : col_start(0)
        , lunique_start(0)
        , lunique_p(0)
        , lunique_n(0)
    {
    }

    /**
     * @brief LabelDataThread constructor
     */
    LabelDataThread(uint in_col_start, uint in_lunique_start, uint in_lunique_p, uint in_lunique_n)
        : col_start(in_col_start)
        , lunique_start(in_lunique_start)
        , lunique_p(in_lunique_p)
        , lunique_n(in_lunique_n)
    {
    }

    uint col_start;
    uint lunique_start;
    uint lunique_p;
    uint lunique_n;
};

/**
 * @brief Performs the connected components labeling (CCL) algorithm assuming 8-connectivity
 *
 * Algorithm is based on "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) variant
 * using decision trees. Kesheng Wu, et al
 * Note: rows are encoded as position in the "rows" array to save lookup times
 *
 * Uses a parallel implementation which divides the input image in horizontal slices (power of 2 size).
 * This function does not perform the final labeling stage. The label map is thus returned using temporary labels.
 * This design decision allows to merge the final labeling stage with the search of max/min sources, performed after this step.
 * For this reason, the array of decision tree is returned by this function to be used in the final labeling stage.
 *
 * @param[in] I (arma::Mat) : Input data matrix
 * @param[in] analysis_thresh_pos (real_t) : Analysis threshold for detection of positive sources
 * @param[in] analysis_thresh_neg (real_t) : Analysis threshold for detection of negative sources
 *
 * @return (std::tuple) Tuple object containing the label map matrix (arma::Mat), the array of decision tree (arma::Mat),
 *                      the mumber of positive labels (uint) and the number of negative labels (uint)
 */
template <bool findNegative = false>
std::tuple<MatStp<int>, MatStp<uint>, uint, uint> labeling_8con(const arma::Mat<real_t>& I, const real_t analysis_thresh_pos, const real_t analysis_thresh_neg)
{
    const size_t cols = I.n_cols;
    const size_t rows = I.n_rows;
#ifndef FFTSHIFT
    const size_t cshift = cols / 2;
    const size_t rshift = rows / 2;
#endif
    assert(cols % 2 == 0);
    assert(rows % 2 == 0);
    assert(cols <= 65536);
    assert(rows <= 65536);

    // Use MapStp because L (label map) shall be initialized with zeroes
    MatStp<int> L(I.n_rows, I.n_cols);

    // A quick and dirty upper bound for the maximimum number of labels.
    const size_t Plength = cols * rows / 2;
    assert(Plength > cols);
    uint Pcols = 1;
    if (findNegative) {
        Pcols = 2;
    }
    // Label equivalence array
    MatStp<uint> P(Plength, Pcols);
    uint* Pp = (uint*)P.colptr(0);
    uint* Pn = (uint*)P.colptr(Pcols - 1);

    // Define maximum number of slices based on the number of available threads
    uint num_slices = tbb::task_scheduler_init::default_num_threads() * 4;

    // Grain size
    size_t grainsize = std::max(int(cols / num_slices), 2);

    // Use this vector to store start column and number of labels assigned by each thread
    tbb::concurrent_vector<LabelDataThread> label_data_per_thread;

    // Scanning phase
    // Use static partitioner because image partitions need to be known in the next step of border merging
    tbb::parallel_for(tbb::blocked_range<size_t>(0, cols, grainsize), [&](const tbb::blocked_range<size_t>& r) {
        const uint lunique_start = (r.begin() * (rows / 2));
        // Init label equivalence array
        Pp[lunique_start] = 0;
        Pn[lunique_start] = 0;
        assert(lunique_start < (cols * rows / 2));
        uint lunique_p = lunique_start + 1;
        uint lunique_n = lunique_start + 1;
        assert(lunique_p >= 0);
        assert(lunique_n >= 0);
        // Start and end columns for this thread
        const uint col_start = r.begin();
        const uint col_end = r.end();

        // Loop over columns
        for (uint c_i = col_start; c_i < col_end; ++c_i) {
            // Set current and previous column indexes
#ifdef FFTSHIFT
            const uint m_c_i = c_i;
            uint m_c_i_prev = (m_c_i == 0) ? m_c_i : m_c_i - 1;
#else
            // If image is shifted, move column by cshift
            const uint m_c_i = c_i >= cshift ? c_i - cshift : c_i + cshift;
            uint m_c_i_prev = (m_c_i == 0) ? cols - 1 : m_c_i - 1;
#endif
            assert(m_c_i < cols);
            assert(m_c_i_prev < cols);

            int* Lcol = L.colptr(m_c_i);
            int* Lcol_prev = L.colptr(m_c_i_prev);
            const real_t* Icol = I.colptr(m_c_i);

            // Indicate whether the left neighbors are valid or not (the first column of each thread should not have left neighbors)
            const bool LeftCol_valid = !(c_i == r.begin());

            // Loop over rows
            for (uint m_r_i = 0; m_r_i < rows; m_r_i++) {
                assert(m_r_i < rows);

                // Positive sources
                if (*(Icol + m_r_i) > analysis_thresh_pos) {
                    const uint m_r_i_prev = (m_r_i == 0) ? m_r_i : m_r_i - 1;
                    int* curL = Lcol + m_r_i;

                    // Get neighbouring pixels
                    const int top_pix = *(Lcol + m_r_i_prev);
                    const int left_pix = *(Lcol_prev + m_r_i);
                    const int topleft_pix = *(Lcol_prev + m_r_i_prev);
                    const uint m_r_i_next = (m_r_i == (rows - 1)) ? m_r_i : m_r_i + 1;
                    const int bottomleft_pix = *(Lcol_prev + m_r_i_next);

                    // These variables indicate what pixels are connected to the current pixel
                    const bool T_left = LeftCol_valid && (left_pix > 0);
#ifdef FFTSHIFT
                    const bool T_top = (m_r_i == 0) ? false : (top_pix > 0);
                    const bool T_topleft = (m_r_i == 0) ? false : (LeftCol_valid && (topleft_pix > 0));
                    const bool T_bottomleft = (m_r_i == (rows - 1)) ? false : (LeftCol_valid && (bottomleft_pix > 0));
#else
                    const bool T_top = (m_r_i == rshift || m_r_i == 0) ? false : (top_pix > 0);
                    const bool T_topleft = (m_r_i == rshift || m_r_i == 0) ? false : (LeftCol_valid && (topleft_pix > 0));
                    const bool T_bottomleft = (m_r_i == (rshift - 1) || m_r_i == (rows - 1)) ? false : (LeftCol_valid && (bottomleft_pix > 0));
#endif
                    if (T_left) {
                        // copy(left)
                        *curL = left_pix;
                    } else {
                        if (T_bottomleft) {
                            if (T_topleft) {
                                // copy(topleft, bottomleft)
                                *curL = set_union(Pp, topleft_pix, bottomleft_pix);
                            } else {
                                if (T_top) {
                                    // copy(top, bottomleft)
                                    *curL = set_union(Pp, top_pix, bottomleft_pix);
                                } else {
                                    // copy(bottomleft)
                                    *curL = bottomleft_pix;
                                }
                            }
                        } else {
                            if (T_topleft) {
                                // copy(topleft)
                                *curL = topleft_pix;
                            } else {
                                if (T_top) {
                                    // copy(top)
                                    *curL = top_pix;
                                } else {
                                    // new label
                                    *curL = lunique_p;
                                    Pp[lunique_p] = lunique_p;
                                    lunique_p = lunique_p + 1;
                                }
                            }
                        }
                    }

                } else {
                    // Negative sources
                    if (findNegative && (*(Icol + m_r_i) < analysis_thresh_neg)) {

                        const uint m_r_i_prev = (m_r_i == 0) ? m_r_i : m_r_i - 1;
                        int* curL = Lcol + m_r_i;

                        // Get neighbouring pixels
                        const int top_pix = *(Lcol + m_r_i_prev);
                        const int left_pix = *(Lcol_prev + m_r_i);
                        const int topleft_pix = *(Lcol_prev + m_r_i_prev);
                        const uint m_r_i_next = (m_r_i == (rows - 1)) ? m_r_i : m_r_i + 1;
                        const int bottomleft_pix = *(Lcol_prev + m_r_i_next);

                        // These variables indicate what pixels are connected to the current pixel
                        const bool T_left = LeftCol_valid && (left_pix < 0);
#ifdef FFTSHIFT
                        const bool T_top = (m_r_i == 0) ? false : (top_pix < 0);
                        const bool T_topleft = (m_r_i == 0) ? false : (LeftCol_valid && (topleft_pix < 0));
                        const bool T_bottomleft = (m_r_i == (rows - 1)) ? false : (LeftCol_valid && (bottomleft_pix < 0));
#else
                        const bool T_top = (m_r_i == rshift || m_r_i == 0) ? false : (top_pix < 0);
                        const bool T_topleft = (m_r_i == rshift || m_r_i == 0) ? false : (LeftCol_valid && (topleft_pix < 0));
                        const bool T_bottomleft = (m_r_i == (rshift - 1) || m_r_i == (rows - 1)) ? false : (LeftCol_valid && (bottomleft_pix < 0));
#endif

                        if (T_left) {
                            // copy(left)
                            *curL = left_pix;
                        } else {
                            if (T_bottomleft) {
                                if (T_topleft) {
                                    // copy(topleft, bottomleft)
                                    *curL = set_union(Pn, -topleft_pix, -bottomleft_pix) * (-1);
                                } else {
                                    if (T_top) {
                                        // copy(top, bottomleft)
                                        *curL = set_union(Pn, -top_pix, -bottomleft_pix) * (-1);
                                    } else {
                                        // copy(bottomleft)
                                        *curL = bottomleft_pix;
                                    }
                                }
                            } else {
                                if (T_topleft) {
                                    // copy(topleft)
                                    *curL = topleft_pix;
                                } else {
                                    if (T_top) {
                                        // copy(top)
                                        *curL = top_pix;
                                    } else {
                                        // new label
                                        *curL = -lunique_n;
                                        Pn[lunique_n] = lunique_n;
                                        lunique_n = lunique_n + 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

#ifndef FFTSHIFT
            // ROW BORDER MERGING (top and bottom matrix margins)
            // Performs border merging between the first and last rows of the matrix
            const uint m_r_i = 0;
            const uint m_r_i_prev = rows - 1;
            int* curL = Lcol + m_r_i;

            // Positive sources
            if (*curL > 0) {
                // Get neighbouring pixels
                const int top_pix = *(Lcol + m_r_i_prev);
                const int topleft_pix = *(Lcol_prev + m_r_i_prev);

                if (top_pix > 0) {
                    *curL = set_union(Pp, *curL, top_pix);
                }
                if ((topleft_pix > 0) && (top_pix != topleft_pix)) {
                    *curL = set_union(Pp, *curL, topleft_pix);
                }
            } else {
                // Negative sources
                if (findNegative && (*curL < 0)) {
                    // Get neighbouring pixels
                    const int top_pix = *(Lcol + m_r_i_prev);
                    const int topleft_pix = *(Lcol_prev + m_r_i_prev);

                    if (top_pix < 0) {
                        *curL = set_union(Pn, -(*curL), -top_pix) * (-1);
                    }
                    if ((topleft_pix < 0) && (top_pix != topleft_pix)) {
                        *curL = set_union(Pn, -(*curL), -topleft_pix) * (-1);
                    }
                }
            }
#endif
        }
        label_data_per_thread.push_back(LabelDataThread(col_start, lunique_start, lunique_p, lunique_n));
    },
        tbb::static_partitioner());

    TIMESTAMP_CCL

    // Sort label_data_per_thread
    std::sort(label_data_per_thread.begin(), label_data_per_thread.end(), [&](const LabelDataThread& a, const LabelDataThread& b) {
        return a.col_start < b.col_start;
    });

    // COLUMN BORDER MERGING
    // Merges the matrix left and right borders as well as all the partitioned regions required for parallel processing
    for (auto&& r : label_data_per_thread) {
        uint c_i = r.col_start;
#ifdef FFTSHIFT
        const uint m_c_i = c_i;
        uint m_c_i_prev = m_c_i == 0 ? m_c_i : m_c_i - 1;
#else
        const uint m_c_i = c_i >= cshift ? c_i - cshift : c_i + cshift;
        uint m_c_i_prev = m_c_i == 0 ? cols - 1 : m_c_i - 1;
#endif
        assert(m_c_i < cols);
        assert(m_c_i_prev < cols);

        int* const Lcol = L.colptr(m_c_i);
        int* const Lcol_prev = L.colptr(m_c_i_prev);

        const bool LeftCol_valid = c_i != 0;

        // Loop over rows
        for (uint m_r_i = 0; m_r_i < rows; m_r_i++) {
            int* curL = Lcol + m_r_i;
            const int cur_pix = *curL;

            // Positive sources
            if (cur_pix > 0) {
#ifdef FFTSHIFT
                const uint m_r_i_prev = (m_r_i == 0) ? m_r_i : m_r_i - 1;
                const uint m_r_i_next = (m_r_i == (rows - 1)) ? m_r_i : m_r_i + 1;
#else
                const uint m_r_i_prev = (m_r_i == 0) ? rows - 1 : (m_r_i == rshift ? m_r_i : m_r_i - 1);
                const uint m_r_i_next = (m_r_i == (rows - 1)) ? 0 : (m_r_i == (rshift - 1) ? m_r_i : m_r_i + 1);
#endif
                // Get neighbouring pixels
                const int topleft_pix = *(Lcol_prev + m_r_i_prev);
                const int left_pix = *(Lcol_prev + m_r_i);
                const int bottomleft_pix = *(Lcol_prev + m_r_i_next);

                // These variables indicate what pixels are connected to the current pixel
                const bool T_left = LeftCol_valid && (left_pix > 0);
#ifdef FFTSHIFT
                const bool T_topleft = (m_r_i == 0) ? false : (LeftCol_valid && (topleft_pix > 0));
                const bool T_bottomleft = (m_r_i == (rows - 1)) ? false : (LeftCol_valid && (bottomleft_pix > 0));
#else
                const bool T_topleft = (m_r_i == rshift) ? false : (LeftCol_valid && (topleft_pix > 0));
                const bool T_bottomleft = (m_r_i == (rshift - 1)) ? false : (LeftCol_valid && (bottomleft_pix > 0));
#endif
                if (T_topleft) {
                    *curL = set_union(Pp, cur_pix, topleft_pix);
                }
                if (T_left) {
                    *curL = set_union(Pp, cur_pix, left_pix);
                }
                if (T_bottomleft) {
                    *curL = set_union(Pp, cur_pix, bottomleft_pix);
                }
            } else {
                // Negative sources
                if (findNegative && (cur_pix < 0)) {
#ifdef FFTSHIFT
                    const uint m_r_i_prev = (m_r_i == 0) ? m_r_i : m_r_i - 1;
                    const uint m_r_i_next = (m_r_i == (rows - 1)) ? m_r_i : m_r_i + 1;
#else
                    const uint m_r_i_prev = (m_r_i == 0) ? rows - 1 : (m_r_i == rshift ? m_r_i : m_r_i - 1);
                    const uint m_r_i_next = (m_r_i == (rows - 1)) ? 0 : (m_r_i == (rshift - 1) ? m_r_i : m_r_i + 1);
#endif
                    // Get neighbouring pixels
                    const int topleft_pix = *(Lcol_prev + m_r_i_prev);
                    const int left_pix = *(Lcol_prev + m_r_i);
                    const int bottomleft_pix = *(Lcol_prev + m_r_i_next);

                    // These variables indicate what pixels are connected to the current pixel
                    const bool T_left = LeftCol_valid && (left_pix < 0);
#ifdef FFTSHIFT
                    const bool T_topleft = (m_r_i == 0) ? false : (LeftCol_valid && (topleft_pix < 0));
                    const bool T_bottomleft = (m_r_i == (rows - 1)) ? false : (LeftCol_valid && (bottomleft_pix < 0));
#else
                    const bool T_topleft = (m_r_i == rshift) ? false : (LeftCol_valid && (topleft_pix < 0));
                    const bool T_bottomleft = (m_r_i == (rshift - 1)) ? false : (LeftCol_valid && (bottomleft_pix < 0));
#endif

                    if (T_topleft) {
                        *curL = set_union(Pn, -cur_pix, -topleft_pix) * (-1);
                    }
                    if (T_left) {
                        *curL = set_union(Pn, -cur_pix, -left_pix) * (-1);
                    }
                    if (T_bottomleft) {
                        *curL = set_union(Pn, -cur_pix, -bottomleft_pix) * (-1);
                    }
                }
            }
        }
    }

    TIMESTAMP_CCL

    // Analysis: positive sources
    uint k = 1;
    for (auto&& r : label_data_per_thread) {
        for (uint i = r.lunique_start + 1; i < r.lunique_p; ++i) {
            if (Pp[i] < i) {
                Pp[i] = Pp[Pp[i]];
            } else {
                Pp[i] = k;
                k = k + 1;
            }
        }
    }
    const uint num_l_pos = k - 1;

    // Analysis: negative sources
    k = 1;
    if (findNegative) {
        for (auto&& r : label_data_per_thread) {
            for (uint i = r.lunique_start + 1; i < r.lunique_n; ++i) {
                if (Pn[i] < i) {
                    Pn[i] = Pn[Pn[i]];
                } else {
                    Pn[i] = k;
                    k = k + 1;
                }
            }
        }
    }
    const uint num_l_neg = k - 1;

    // Return label map (temporary labels), array of decision tree, number of positive and negative labels
    return std::make_tuple(std::move(L), std::move(P), num_l_pos, num_l_neg);
}

/**
 * @brief Performs the connected components labeling (CCL) algorithm assuming 4-connectivity
 *
 * Algorithm is based on "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) variant
 * using decision trees. Kesheng Wu, et al
 * Note: rows are encoded as position in the "rows" array to save lookup times
 *
 * Uses a parallel implementation which divides the input image in horizontal slices (power of 2 size).
 * This function does not perform the final labeling stage. The label map is thus returned using temporary labels.
 * This design decision allows to merge the final labeling stage with the search of max/min sources, performed after this step.
 * For this reason, the array of decision tree is returned by this function to be used in the final labeling stage.
 *
 * @param[in] I (arma::Mat) : Input data matrix
 * @param[in] analysis_thresh_pos (real_t) : Analysis threshold for detection of positive sources
 * @param[in] analysis_thresh_neg (real_t) : Analysis threshold for detection of negative sources
 *
 * @return (std::tuple) Tuple object containing the label map matrix (arma::Mat), the array of decision tree (arma::Mat),
 *                      the mumber of positive labels (uint) and the number of negative labels (uint)
 */
template <bool findNegative>
std::tuple<MatStp<int>, MatStp<uint>, uint, uint> labeling_4con(const arma::Mat<real_t>& I, const real_t analysis_thresh_pos, const real_t analysis_thresh_neg)
{
    const size_t cols = I.n_cols;
    const size_t rows = I.n_rows;
#ifndef FFTSHIFT
    const size_t cshift = cols / 2;
    const size_t rshift = rows / 2;
#endif
    assert(cols % 2 == 0);
    assert(rows % 2 == 0);
    assert(cols <= 65536);
    assert(rows <= 65536);

    // Use MapStp because L (label map) shall be initialized with zeroes
    MatStp<int> L(I.n_rows, I.n_cols);

    // A quick and dirty upper bound for the maximimum number of labels.
    const size_t Plength = cols * rows / 2;
    assert(Plength > cols);
    uint Pcols = 1;
    if (findNegative) {
        Pcols = 2;
    }
    // Array of decision tree
    MatStp<uint> P(Plength, Pcols);
    uint* Pp = (uint*)P.colptr(0);
    uint* Pn = (uint*)P.colptr(Pcols - 1);

    // Define maximum number of slices based on the number of available threads
    uint num_slices = tbb::task_scheduler_init::default_num_threads() * 4;

    // Grain size
    size_t grainsize = std::max(int(cols / num_slices), 2);

    // Use this vector to store start column and number of labels assigned by each thread
    tbb::concurrent_vector<LabelDataThread> label_data_per_thread;

    // Scanning phase
    // Use static partitioner because image partitions need to be known in the next step of border merging
    tbb::parallel_for(tbb::blocked_range<size_t>(0, cols, grainsize), [&](const tbb::blocked_range<size_t>& r) {
        const uint lunique_start = (r.begin() * (rows / 2));
        // Init label equivalence array
        Pp[lunique_start] = 0;
        Pn[lunique_start] = 0;
        assert(lunique_start < (cols * rows / 2));
        uint lunique_p = lunique_start + 1;
        uint lunique_n = lunique_start + 1;
        assert(lunique_p >= 0);
        assert(lunique_n >= 0);
        // Start and end columns for this thread
        const uint col_start = r.begin();
        const uint col_end = r.end();

        // Loop over cols
        for (uint c_i = col_start; c_i < col_end; ++c_i) {
            // Set current and previous column indexes
#ifdef FFTSHIFT
            const uint m_c_i = c_i;
            uint m_c_i_prev = (m_c_i == 0) ? m_c_i : m_c_i - 1;
#else
            const uint m_c_i = c_i >= cshift ? c_i - cshift : c_i + cshift;
            uint m_c_i_prev = (m_c_i == 0) ? cols - 1 : m_c_i - 1;
#endif
            assert(m_c_i < cols);
            assert(m_c_i_prev < cols);

            int* Lcol = L.colptr(m_c_i);
            int* Lcol_prev = L.colptr(m_c_i_prev);
            const real_t* Icol = I.colptr(m_c_i);

            // Indicate whether the left neighbor is valid or not (the first column of each thread should not have left neighbor)
            const bool LeftCol_valid = !(c_i == r.begin());

            // Loop over rows
            for (uint m_r_i = 0; m_r_i < rows; m_r_i++) {
                assert(m_r_i < rows);

                // Positive sources
                if (*(Icol + m_r_i) > analysis_thresh_pos) {
                    const uint m_r_i_prev = (m_r_i == 0) ? m_r_i : m_r_i - 1;
                    int* curL = Lcol + m_r_i;

                    // Get neighbouring pixels
                    const int top_pix = *(Lcol + m_r_i_prev);
                    const int left_pix = *(Lcol_prev + m_r_i);

                    // These variables indicate what pixels are connected to the current pixel
                    const bool T_left = LeftCol_valid && (left_pix > 0);
#ifdef FFTSHIFT
                    const bool T_top = (m_r_i == 0) ? false : (top_pix > 0);
#else
                    const bool T_top = (m_r_i == rshift || m_r_i == 0) ? false : (top_pix > 0);
#endif

                    if (T_left) {
                        if (T_top) {
                            // copy(top, left)
                            *curL = set_union(Pp, top_pix, left_pix);
                        } else {
                            // copy(left)
                            *curL = left_pix;
                        }
                    } else {
                        if (T_top) {
                            // copy(top)
                            *curL = top_pix;
                        } else {
                            // new label
                            *curL = lunique_p;
                            Pp[lunique_p] = lunique_p;
                            lunique_p = lunique_p + 1;
                        }
                    }
                } else {
                    // Negative sources
                    if (findNegative && (*(Icol + m_r_i) < analysis_thresh_neg)) {
                        const uint m_r_i_prev = (m_r_i == 0) ? m_r_i : m_r_i - 1;
                        int* curL = Lcol + m_r_i;

                        // Get neighbouring pixels
                        const int top_pix = *(Lcol + m_r_i_prev);
                        const int left_pix = *(Lcol_prev + m_r_i);

                        // These variables indicate what pixels are connected to the current pixel
                        const bool T_left = LeftCol_valid && (left_pix < 0);
#ifdef FFTSHIFT
                        const bool T_top = (m_r_i == 0) ? false : (top_pix < 0);
#else
                        const bool T_top = (m_r_i == rshift || m_r_i == 0) ? false : (top_pix < 0);
#endif

                        if (T_left) {
                            if (T_top) {
                                // copy(top, left)
                                *curL = set_union(Pn, -top_pix, -left_pix) * (-1);
                            } else {
                                // copy(left)
                                *curL = left_pix;
                            }
                        } else {
                            if (T_top) {
                                // copy(top)
                                *curL = top_pix;
                            } else {
                                // new label
                                *curL = -lunique_n;
                                Pn[lunique_n] = lunique_n;
                                lunique_n = lunique_n + 1;
                            }
                        }
                    }
                }
            }

#ifndef FFTSHIFT
            // ROW BORDER MERGING (top and bottom matrix margins)
            // Performs border merging between the first and last rows of the matrix
            const uint m_r_i = 0;
            const uint m_r_i_prev = rows - 1;
            int* curL = Lcol + m_r_i;
            // Positive sources
            if (*curL > 0) {
                // Get neighbouring pixels
                const int top_pix = *(Lcol + m_r_i_prev);
                if (top_pix > 0) {
                    *curL = set_union(Pp, *curL, top_pix);
                }
            } else {
                // Negative sources
                if (findNegative && (*curL < 0)) {
                    // Get neighbouring pixels
                    const int top_pix = *(Lcol + m_r_i_prev);
                    if (top_pix < 0) {
                        *curL = set_union(Pn, -(*curL), -top_pix) * (-1);
                    }
                }
            }
#endif
        }
        label_data_per_thread.push_back(LabelDataThread(col_start, lunique_start, lunique_p, lunique_n));
    },
        tbb::static_partitioner());

    TIMESTAMP_CCL

    // Sort label_data_per_thread
    std::sort(label_data_per_thread.begin(), label_data_per_thread.end(), [&](const LabelDataThread& a, const LabelDataThread& b) {
        return a.col_start < b.col_start;
    });

    // COLUMN BORDER MERGING
    // Merges the matrix left and right borders as well as all the partitioned regions required for parallel processing
    for (auto&& r : label_data_per_thread) {
        uint c_i = r.col_start;
#ifdef FFTSHIFT
        const uint m_c_i = c_i;
        uint m_c_i_prev = m_c_i == 0 ? m_c_i : m_c_i - 1;
#else
        const uint m_c_i = c_i >= cshift ? c_i - cshift : c_i + cshift;
        uint m_c_i_prev = m_c_i == 0 ? cols - 1 : m_c_i - 1;
#endif
        assert(m_c_i < cols);
        assert(m_c_i_prev < cols);

        int* const Lcol = L.colptr(m_c_i);
        int* const Lcol_prev = L.colptr(m_c_i_prev);

        const bool LeftCol_valid = c_i != 0;
        for (uint m_r_i = 0; m_r_i < rows; m_r_i++) {
            int* curL = Lcol + m_r_i;
            const int cur_pix = *curL;
            // Positive sources
            if (cur_pix > 0) {
                // Get neighbouring pixels
                const int left_pix = *(Lcol_prev + m_r_i);

                if (LeftCol_valid && (left_pix > 0)) {
                    *curL = set_union(Pp, cur_pix, left_pix);
                }
            } else {
                // Negative sources
                if (findNegative && (cur_pix < 0)) {
                    // Get neighbouring pixels
                    const int left_pix = *(Lcol_prev + m_r_i);

                    if (LeftCol_valid && (left_pix < 0)) {
                        *curL = set_union(Pn, -cur_pix, -left_pix) * (-1);
                    }
                }
            }
        }
    }

    TIMESTAMP_CCL

    // Analysis: positive sources
    uint k = 1;
    for (auto&& r : label_data_per_thread) {
        for (uint i = r.lunique_start + 1; i < r.lunique_p; ++i) {
            if (Pp[i] < i) {
                Pp[i] = Pp[Pp[i]];
            } else {
                Pp[i] = k;
                k = k + 1;
            }
        }
    }
    const uint num_l_pos = k - 1;

    // Analysis: negative sources
    k = 1;
    if (findNegative) {
        for (auto&& r : label_data_per_thread) {
            for (uint i = r.lunique_start + 1; i < r.lunique_n; ++i) {
                if (Pn[i] < i) {
                    Pn[i] = Pn[Pn[i]];
                } else {
                    Pn[i] = k;
                    k = k + 1;
                }
            }
        }
    }
    const uint num_l_neg = k - 1;

    // Return label map (temporary labels), array of decision tree, number of positive and negative labels
    return std::make_tuple(std::move(L), std::move(P), num_l_pos, num_l_neg);
}
}

#endif /* CCL_H */
