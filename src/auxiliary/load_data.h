#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include <armadillo>
#include <cnpy.h>
#include <memory>

/**
*   @brief Loads a complex array
*
*   Reads an npy struct and loads the data array into an armadillo complex matrix
*
*   @param[in] npy array that contains the data to be read
*
*   @return Armadillo matrix with npy values
*/
arma::cx_mat load_npy_complex_array(cnpy::NpyArray& npy) throw(std::invalid_argument);

/**
*   @brief Loads a double array
*
*   Reads an npy struct and loads the data array into an armadillo double matrix
*
*   @param[in] npy array that contains the data to be read
*
*   @return Armadillo matrix with npy values
*/
arma::mat load_npy_double_array(cnpy::NpyArray& npy) throw(std::invalid_argument);

#endif /* LOAD_DATA_H */
