#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include <armadillo>
#include <cnpy.h>
#include <memory>

/**
*   @brief Loads the uvw array
*
*   Reads an npy struct and loads the data array into an armadillo matrix
*
*   @param[in] npy npy array that contains the data to be read
*
*   @return Armadillo matrix with npy values
*/
arma::cx_mat load_npy_array(cnpy::NpyArray* npy) throw(std::invalid_argument);

#endif /* LOAD_DATA_H */
