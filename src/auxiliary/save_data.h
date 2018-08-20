#ifndef SAVE_DATA_H
#define SAVE_DATA_H

#include <armadillo>
#include <cnpy.h>

/**
*   @brief Save an armadillo matrix in NPZ format
*
*   @param[in] zipname (string) output NPZ file name
*   @param[in] fname (string) matrix name within NPZ file
*   @param[in] data (arma::Mat<T>) armadillo matrix that contains the data to be saved
*   @param[in] mode (string) file saving mode. Default: "w" creates new file. Use "a" to append to existing file.
*
*/
template <typename T>
void npz_save(const std::string zipname, const std::string fname, arma::Mat<T>& data, std::string mode = "w")
{
    arma::Mat<T> tdata = data.st(); // transpose is required because npz saves in C memory order (fortran_order = false)
    std::vector<size_t> shape = { tdata.n_cols, tdata.n_rows };
    cnpy::npz_save(zipname, fname, tdata.memptr(), shape, mode);
}

#endif /* SAVE_DATA_H */
