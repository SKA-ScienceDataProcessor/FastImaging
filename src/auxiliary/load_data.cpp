#include "load_data.h"

arma::cx_mat load_npy_array(cnpy::NpyArray* npy) throw(std::invalid_argument)
{
    // Gets the uvw size
    unsigned int npy_size(npy->shape[0]);
    for (size_t i(1); i < npy->shape.size(); i++) {

        npy_size = npy_size * npy->shape[i];
    }

    unsigned int npy_rows = npy->shape[0];
    unsigned int npy_cols;
    if (npy->shape.size() < 2) {
        npy_cols = 1;
    } else {
        npy_cols = npy->shape[1];
    }

    // Reads the uvw data

    // Complex values
    if (npy->word_size == 16) {

        std::unique_ptr<std::complex<double> > arr = std::unique_ptr<std::complex<double> >(reinterpret_cast<std::complex<double>*>(npy->data));

        // Creates the armadillo matrices
        arma::cx_mat dataMat(arr.get(), npy_rows, npy_cols);
        dataMat.set_size(npy_rows, npy_cols);

        return dataMat;
    }
    // Float values
    else if (npy->word_size == 8) {

        std::unique_ptr<double> arr = std::unique_ptr<double>((double*)(npy->data));

        // Creates the armadillo matrices
        arma::mat realData(arr.get(), npy_rows, npy_cols);
        arma::mat imagData(realData.n_rows, realData.n_cols, arma::fill::zeros);

        arma::cx_mat dataMat(realData, imagData);
        dataMat.set_size(npy_rows, npy_cols);

        return dataMat;
    }

    throw std::invalid_argument("Invalid data type. Must be float or complex!");
}
