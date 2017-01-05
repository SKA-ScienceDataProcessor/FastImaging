#include "load_data.h"

arma::mat load_npy_double_array(cnpy::NpyArray& npy) throw(std::invalid_argument)
{
    // Gets the uvw size
    unsigned int npy_size(npy.shape[0]);
    for (size_t i(1); i < npy.shape.size(); i++) {
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

        std::unique_ptr<double> arr = std::unique_ptr<double>((double*)(npy.data));

        // Creates the armadillo matrices
        arma::mat dataMat(arr.get(), npy_rows, npy_cols);
        dataMat.set_size(npy_rows, npy_cols);

        if (!npy.fortran_order) {
            dataMat = dataMat.st();
        }

        return dataMat;
    }

    throw std::invalid_argument("Invalid data type. Must be float!");
}

arma::cx_mat load_npy_complex_array(cnpy::NpyArray& npy) throw(std::invalid_argument)
{
    // Gets the uvw size
    unsigned int npy_size(npy.shape[0]);
    for (size_t i(1); i < npy.shape.size(); i++) {

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

        std::unique_ptr<std::complex<double> > arr = std::unique_ptr<std::complex<double> >(reinterpret_cast<std::complex<double>*>(npy.data));

        // Creates the armadillo matrices
        arma::cx_mat dataMat(arr.get(), npy_rows, npy_cols);
        dataMat.set_size(npy_rows, npy_cols);

        if (!npy.fortran_order) {
            dataMat = dataMat.st();
        }

        return dataMat;
    }
    // Float values
    else if (npy.word_size == 8) {

        std::unique_ptr<double> arr = std::unique_ptr<double>((double*)(npy.data));

        // Creates the armadillo matrices
        arma::mat realData(arr.get(), npy_rows, npy_cols);
        arma::mat imagData(realData.n_rows, realData.n_cols, arma::fill::zeros);

        arma::cx_mat dataMat(realData, imagData);
        dataMat.set_size(npy_rows, npy_cols);

        if (!npy.fortran_order) {
            dataMat = dataMat.st();
        }

        return dataMat;
    }

    throw std::invalid_argument("Invalid data type. Must be float or complex!");
}
