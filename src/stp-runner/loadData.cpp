#include "stp-runner.h"

/**
*   @brief Loads the uvw array
*
*   Reads an npy struct and loads the data array into an armadillo complex matrix
*
*   @param[in] npy npy array that contains the data to be read
*
*   @return Armadillo complex mat with npy values
*/
template <typename data>
cx_mat loadComplexNpyArray(NpyArray* npy)
{ 
     _logger -> debug("Starting loadNpyArray function");

    //Gets the uvw size
    _logger -> debug("Calculating npy array size");
    unsigned int npy_size(npy->shape[0]);
    for(size_t i=1; i<npy->shape.size(); i++)
    {
        npy_size = npy_size * npy->shape[i];
    }

    //Creates the armadillo matrices
    _logger -> debug("Creating armadillo matrices");
    cx_mat dataMat;

    //Reads the uvw data
    _logger -> debug("Reading npy array data");
    data* arr((data*) npy->data);
    for(int i=0; i<npy_size; i++)
    {
       dataMat << arr[i];
    }

    //Delete data
    delete[] arr;

    return dataMat;
}

/**
*   @brief Loads the uvw array
*
*   Reads an npy struct and loads the data array into an armadillo matrix
*
*   @param[in] npy npy array that contains the data to be read
*
*   @return Armadillo matrix with npy values
*/
mat loadNpyArray(NpyArray* npy)
{ 
     _logger -> debug("Starting loadNpyArray function");

    //Gets the uvw size
    _logger -> debug("Calculating npy array size");
    unsigned int npy_size(npy->shape[0]);
    for(size_t i=1; i<npy->shape.size(); i++)
    {
        npy_size = npy_size * npy->shape[i];
    }

    //Reads the uvw data
    _logger -> debug("Reading npy array data");
    double* arr((double*) npy->data);

    //Creates the armadillo matrices
    _logger -> debug("Creating armadillo matrices");
    mat dataMat( (double*) arr, npy_size, 1);

    //Delete data
    delete[] arr;

    return dataMat;
}
