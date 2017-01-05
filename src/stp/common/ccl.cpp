/*M///////////////////////////////////////////////////////////////////////////////////////
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
//
///////////////////////////////////////////////////////////////////////////////////////M*/

#include "ccl.h"

namespace stp {

// Find the root of the tree of node i.
inline static arma::sword find_root(const arma::sword* P, arma::sword i)
{
    arma::sword root = i;
    while (P[root] < root) {
        root = P[root];
    }
    return root;
}

// Make all nodes in the path of node i point to root.
inline static void set_root(arma::sword* P, arma::sword i, arma::sword root)
{
    while (P[i] < i) {
        arma::sword j = P[i];
        P[i] = root;
        i = j;
    }
    P[i] = root;
}

// Find the root of the tree of the node i and compress the path in the process.
inline static arma::sword find(arma::sword* P, arma::sword i)
{
    arma::sword root = find_root(P, i);
    set_root(P, i, root);
    return root;
}

// Unite the two trees containing nodes i and j and return the new root.
inline static arma::sword set_union(arma::sword* P, arma::sword i, arma::sword j)
{
    arma::sword root = find_root(P, i);
    if (i != j) {
        arma::sword rootj = find_root(P, j);
        if (root > rootj) {
            root = rootj;
        }
        set_root(P, j, root);
    }
    set_root(P, i, root);
    return root;
}

// Flatten the Union Find tree and relabel the components.
inline static arma::sword flattenL(arma::sword* P, arma::sword length)
{
    arma::sword k = 1;
    for (arma::sword i = 1; i < length; ++i) {
        if (P[i] < i) {
            P[i] = P[P[i]];
        } else {
            P[i] = k;
            k = k + 1;
        }
    }
    return k;
}

arma::sword labeling(const arma::imat& I, arma::imat& L)
{
    const int rows = L.n_cols;
    const int cols = L.n_rows;

    // A quick and dirty upper bound for the maximimum number of labels.  The 4 comes from
    // the fact that a 3x3 block can never have more than 4 unique labels for both 4 & 8-way
    const size_t Plength = 4 * (size_t(rows + 3 - 1) / 3) * (size_t(cols + 3 - 1) / 3);
    arma::sword* P = (arma::sword*)malloc(sizeof(arma::sword) * Plength);
    P[0] = 0;
    arma::sword lunique = 1;
    // scanning phase
    for (int r_i = 0; r_i < rows; ++r_i) {
        arma::sword* const Lrow = L.colptr(r_i);
        arma::sword* const Lrow_prev = L.colptr(std::max(r_i - 1, 0));
        const arma::sword* const Irow = I.colptr(r_i);
        const arma::sword* const Irow_prev = I.colptr(std::max(r_i - 1, 0));
        arma::sword* Lrows[2] = {
            Lrow,
            Lrow_prev
        };
        const arma::sword* Irows[2] = {
            Irow,
            Irow_prev
        };

        // B & D only
        const int b = 0;
        const int d = 1;
        const bool T_b_r = (r_i - G4[b][0]) >= 0;
        for (int c_i = 0; Irows[0] != Irow + cols; ++Irows[0], c_i++) {
            if (!*Irows[0]) {
                Lrow[c_i] = 0;
                continue;
            }
            Irows[1] = Irow_prev + c_i;
            Lrows[0] = Lrow + c_i;
            Lrows[1] = Lrow_prev + c_i;
            const bool T_b = T_b_r && *(Irows[G4[b][0]] + G4[b][1]);
            const bool T_d = (c_i + G4[d][1]) >= 0 && *(Irows[G4[d][0]] + G4[d][1]);
            if (T_b) {
                if (T_d) {
                    // copy(d, b)
                    *Lrows[0] = set_union(P, *(Lrows[G4[d][0]] + G4[d][1]), *(Lrows[G4[b][0]] + G4[b][1]));
                } else {
                    // copy(b)
                    *Lrows[0] = *(Lrows[G4[b][0]] + G4[b][1]);
                }
            } else {
                if (T_d) {
                    // copy(d)
                    *Lrows[0] = *(Lrows[G4[d][0]] + G4[d][1]);
                } else {
                    // new label
                    *Lrows[0] = lunique;
                    P[lunique] = lunique;
                    lunique = lunique + 1;
                }
            }
        }
    }

    // analysis
    arma::sword nLabels = flattenL(P, lunique);

    for (int r_i = 0; r_i < rows; ++r_i) {
        arma::sword* Lrow_start = L.colptr(r_i);
        arma::sword* Lrow_end = Lrow_start + cols;
        arma::sword* Lrow = Lrow_start;
        for (int c_i = 0; Lrow != Lrow_end; ++Lrow, ++c_i) {
            const arma::sword l = P[*Lrow];
            *Lrow = l;
        }
    }

    free(P);

    return nLabels - 1;
}
}
