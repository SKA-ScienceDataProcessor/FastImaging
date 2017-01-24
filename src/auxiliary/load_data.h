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

/**
*   @brief Loads simulation data from NPZ file into armadillo matrices
*
*   Reads an NPZ file with uvw_lambda, model and vis arrays and loads these arrays into distinct armadillo matrices
*
*   @param[in] file_path (string): Input filename with simulation data to be read
*   @param[out] input_uvw (arma::mat): Armadillo matrix with input uvw lambda values
*   @param[out] input_model (arma::cx_mat): Armadillo matrix with input model values
*   @param[out] input_vis (arma::cx_mat): Armadillo matrix with input vis values
*
*/
void load_npz_simdata(std::string file_path, arma::mat& input_uvw, arma::cx_mat& input_model, arma::cx_mat& input_vis);

#endif /* LOAD_DATA_H */
