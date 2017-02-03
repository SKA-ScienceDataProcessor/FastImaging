#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include <armadillo>
#include <cnpy.h>
#include <memory>

/**
*   @brief Loads a complex array from NPZ or NPY file
*
*   Loads a matrix data array from a NPZ or NPY file into an armadillo complex matrix.
*   If var_name is empty, NPY file is assumed. Otherwise reads var_name matrix from a NPZ file.
*
*   @param[in] file_path (string) npz file path that contains the numpy data
*   @param[in] var_name (string) variable name of matrix data to be read
*
*   @return Armadillo complex matrix with npy values
*/
arma::cx_mat load_npy_complex_array(std::string file_path, std::string var_name = std::string()) throw(std::invalid_argument);

/**
*   @brief Loads a double array from NPZ or NPY file
*
*   Loads a matrix data array from a NPZ or NPY file into an armadillo double matrix.
*   If var_name is empty, NPY file is assumed. Otherwise reads var_name matrix from a NPZ file.
*
*   @param[in] file_path (string) npz file path that contains the numpy data
*   @param[in] var_name (string) variable name of matrix data to be read
*
*   @return Armadillo matrix with npy values
*/
arma::mat load_npy_double_array(std::string file_path, std::string var_name = std::string()) throw(std::invalid_argument);

#endif /* LOAD_DATA_H */
