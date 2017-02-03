/**
* @file stp_python.h
* Contains the prototypes of stp python wrapper functions
*/

#ifndef STP_PYTHON_H
#define STP_PYTHON_H

#include <armadillo>
#include <complex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace stp_python {

/**
 * @brief The KernelFunction enum
 */
enum struct KernelFunction {
    TopHat = 0,
    Triangle,
    Sinc,
    Gaussian,
    GaussianSinc
};

using np_complex_array = pybind11::array_t<std::complex<double>, pybind11::array::f_style | pybind11::array::forcecast>; // set fortran_style (motivated by armadillo matrices)
using np_double_array = pybind11::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>; // set fortran_style (motivated by armadillo matrices)

/**
 * @brief Convenience wrapper over image_visibilities function.
 *
 * This function shall be called from python code.
 *
 * @param vis (numpy.ndarray<np.complex_>): Complex visibilities. 1D array, shape: (n_vis,).
 * @param uvw_lambda (numpy.ndarray<np.float_>): UVW-coordinates of visibilities. Units are multiples of wavelength.
 *                                        2D array, shape: (n_vis, 3). Assumed ordering is u,v,w.
 * @param image_size (int): Width of the image in pixels. Assumes (image_size/2, image_size/2) corresponds to the origin in UV-space.
 * @param cell_size (double): Angular-width of a synthesized pixel in the image to be created (arcsecond).
 * @param kernel_func (KernelFunction): Choice of kernel function from limited selection (see KernelFunction enum structure).
 *                                           Default = KernelFunction::GaussianSinc.
 * @param kernel_trunc_radius (double): Truncation radius of the kernel to be used.
 *                                      Default = 3.0.
 * @param kernel_support (int): Defines the 'radius' of the bounding box within which convolution takes place (also known as half-support).
 *                              Box width in pixels = 2*support + 1. The central pixel is the one nearest to the UV co-ordinates.
 *                              Default = 3.
 * @param kernel_exact (bool): If true, calculates exact kernel values for every UV-sample. Otherwise, oversampling is used.
 *                             Default = true.
 * @param kernel_oversampling (int): Controls kernel-generation if 'kernel_exact == False'. Larger values give a finer-sampled set of pre-cached kernels.
 *                                   Default = 9.
 * @param normalize (bool): Whether or not the returned image and beam should be normalized such that the beam peaks at a value of 1.0 Jansky.
 *                          You normally want this to be true, but it may be interesting to check the raw values for debugging purposes.
 *                          Default = true.
 *
 * @return (pybind11::tuple): Tuple of numpy.ndarrays representing the image map and beam model (image, beam).
 */
pybind11::tuple image_visibilities_wrapper(
    np_complex_array vis, // numpy.ndarray<np.complex_>
    np_double_array uvw_lambda, // numpy.ndarray<np.float_>
    int image_size, // int
    double cell_size, // double
    KernelFunction kernel_func = KernelFunction::GaussianSinc, // enum
    double kernel_trunc_radius = 3.0, // double
    int kernel_support = 3, // int
    bool kernel_exact = true, // bool
    int kernel_oversampling = 9, // int
    bool normalize = true); // bool

/**
 * @brief Convenience wrapper over source_find_image function.
 *
 * This function shall be called from python code.
 *
 * @param image_data (numpy.ndarray<np.float_>): Real component of iFFT'd image data (typically uses image returned by image_visibilities function).
 * @param detection_n_sigma (double): Detection threshold as multiple of RMS.
 * @param analysis_n_sigma (double): Analysis threshold as multiple of RMS.
 * @param rms_est (double): RMS estimate (may be 0.0, in which case RMS is estimated from the image data).
 *                          Default = 0.0.
 *
 * @return (pybind11::list): List of tuples representing the source-detections.
 *                           Tuple components are as follows: (sign, val, x_idx, y_idx, xbar, ybar), where:
 *                             - 'sign' is +1 or -1 (int), representing whether the source is positive or negative;
 *                             - 'val' (double) is the 'extremum_val', i.e. max or min pixel value for the positive or negative source case;
 *                             - 'x_idx,y_idx' (int) are the pixel-index of the extremum value;
 *                             - 'xbar, ybar' (double) are 'centre-of-mass' locations for the source-detection island.
 */
std::vector<std::tuple<int, double, int, int, double, double> > source_find_wrapper(
    np_double_array image_data, // numpy.ndarray<np.float_>
    double detection_n_sigma, // double
    double analysis_n_sigma, // double
    double rms_est = 0.0); // double
}

#endif /* STP_PYTHON_H */
