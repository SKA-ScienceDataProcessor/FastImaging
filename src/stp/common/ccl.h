#ifndef CCL_H
#define CCL_H

#include <armadillo>

namespace stp {

// reference for 4-way: {{-1, 0}, {0, -1}};//b, d neighborhoods
const int G4[2][2] = { { 1, 0 }, { 0, -1 } }; //b, d neighborhoods

/**
 * @brief Performs the connected components labeling (CCL) algorithm using 4-connectivity
 *
 * Based on "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) variant
 * using decision trees. Kesheng Wu, et al
 * Note: rows are encoded as position in the "rows" array to save lookup times
 *
 * Receives an binary matrix (arma::umat) and returns label map as arma::umat
 *
 * @param[in] I (arma::umat) : Binary input matrix
 * @param[in] L (arma::umat) : Label map matrix
 *
 * @return Number of labels
 */
int labeling(const arma::Mat<char>& I, arma::Mat<int>& L);
}

#endif /* CCL_H */
