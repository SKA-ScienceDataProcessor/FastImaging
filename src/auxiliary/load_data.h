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
template <typename T>
arma::Mat<std::complex<T>> load_npy_complex_array(const std::string& file_path, const std::string& var_name = std::string())
{
    cnpy::NpyArray npy;
    if (var_name.empty()) {
        npy = cnpy::npy_load(file_path);
    } else {
        npy = cnpy::npz_load(file_path, var_name);
    }

    // Gets the uvw size
    unsigned int npy_size(npy.shape[0]);
    for (size_t i = 1; i < npy.shape.size(); i++) {

        npy_size = npy_size * npy.shape[i];
    }

    unsigned int npy_rows;
    unsigned int npy_cols;

    if (npy.fortran_order) {
        npy_rows = npy.shape[0];
        if (npy.shape.size() < 2) {
            npy_cols = 1;
        } else {
            npy_cols = npy.shape[1];
        }
    } else {
        npy_cols = npy.shape[0];
        if (npy.shape.size() < 2) {
            npy_rows = 1;
        } else {
            npy_rows = npy.shape[1];
        }
    }

    // Complex values
    if (npy.word_size == 16) {

        // Creates the armadillo matrices
        arma::cx_mat dataMat(npy.data<std::complex<double>>(), npy_rows, npy_cols);
        if (!npy.fortran_order) {
            dataMat = dataMat.st();
        }

        return arma::conv_to<arma::Mat<std::complex<T>>>::from(dataMat);
    }
    // Float values
    else if (npy.word_size == 8) {

        // Creates the armadillo matrices
        arma::Mat<std::complex<float>> dataMat(npy.data<std::complex<float>>(), npy_rows, npy_cols);
        if (!npy.fortran_order) {
            dataMat = dataMat.st();
        }

        return arma::conv_to<arma::Mat<std::complex<T>>>::from(dataMat);
    }

    throw std::invalid_argument("Invalid data type. Must be float or complex!");
}
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
template <typename T>
arma::Mat<T> load_npy_double_array(const std::string& file_path, const std::string& var_name = std::string())
{
    cnpy::NpyArray npy;
    if (var_name.empty()) {
        npy = cnpy::npy_load(file_path);
    } else {
        npy = cnpy::npz_load(file_path, var_name);
    }

    // Gets the uvw size
    unsigned int npy_size(npy.shape[0]);
    for (size_t i = 1; i < npy.shape.size(); i++) {
        npy_size = npy_size * npy.shape[i];
    }

    unsigned int npy_rows;
    unsigned int npy_cols;

    if (npy.fortran_order) {
        npy_rows = npy.shape[0];
        if (npy.shape.size() < 2) {
            npy_cols = 1;
        } else {
            npy_cols = npy.shape[1];
        }
    } else {
        npy_cols = npy.shape[0];
        if (npy.shape.size() < 2) {
            npy_rows = 1;
        } else {
            npy_rows = npy.shape[1];
        }
    }

    if (npy.word_size == 8) {

        // Creates the armadillo matrices
        arma::mat dataMat((double*)(npy.data<double>()), npy_rows, npy_cols);
        if (!npy.fortran_order) {
            dataMat = dataMat.st();
        }

        return arma::conv_to<arma::Mat<T>>::from(dataMat);
    }

    if (npy.word_size == 4) {

        // Creates the armadillo matrices
        arma::Mat<float> dataMat((float*)(npy.data<float>()), npy_rows, npy_cols);
        if (!npy.fortran_order) {
            dataMat = dataMat.st();
        }

        return arma::conv_to<arma::Mat<T>>::from(dataMat);
    }

    throw std::invalid_argument("Invalid data type. Must be float!");
}

#endif /* LOAD_DATA_H */
