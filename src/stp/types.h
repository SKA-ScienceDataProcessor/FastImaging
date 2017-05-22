/** @file types.h
 *  @brief Include of types
 *
 *  @bug No known bugs.
 */

#ifndef TYPES_H
#define TYPES_H

#include <complex>

#ifdef USE_FLOAT
using real_t = float;
using cx_real_t = std::complex<float>;
#else
using real_t = double;
using cx_real_t = std::complex<double>;
#endif

#endif /* TYPES_H */
