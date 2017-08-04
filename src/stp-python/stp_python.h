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
#include <stp.h>

namespace stp_python {

using np_complex_double_array = pybind11::array_t<std::complex<double>, pybind11::array::f_style | pybind11::array::forcecast>; // set fortran_style (motivated by armadillo matrices)
using np_double_array = pybind11::array_t<double, pybind11::array::f_style | pybind11::array::forcecast>; // set fortran_style (motivated by armadillo matrices)
using np_complex_real_array = pybind11::array_t<std::complex<real_t>, pybind11::array::f_style | pybind11::array::forcecast>; // set fortran_style (motivated by armadillo matrices)
using np_real_array = pybind11::array_t<real_t, pybind11::array::f_style | pybind11::array::forcecast>; // set fortran_style (motivated by armadillo matrices)

/**
 * @brief Convenience wrapper over image_visibilities function.
 *
 * This function shall be called from python code.
 *
 * @param[in] vis (numpy.ndarray<np.complex_>): Complex visibilities. 1D array, shape: (n_vis,).
 * @param[in] snr_weights (numpy.ndarray<np.float_>): Visibility weights. 1D array, shape: (n_vis,).
 * @param[in] uvw_lambda (numpy.ndarray<np.float_>): UVW-coordinates of visibilities. Units are multiples of wavelength.
 *                                               2D array, shape: (n_vis, 3). Assumed ordering is u,v,w.
 * @param[in] image_size (int): Width of the image in pixels. Assumes (image_size/2, image_size/2) corresponds to the origin in UV-space.
 * @param[in] cell_size (double): Angular-width of a synthesized pixel in the image to be created (arcsecond).
 * @param[in] kernel_func (KernelFunction): Choice of kernel function from limited selection (see KernelFunction enum structure).
 *                                      Default = KernelFunction::GaussianSinc.
 * @param[in] kernel_trunc_radius (double): Truncation radius of the kernel to be used.
 *                                      Default = 3.0.
 * @param[in] kernel_support (int): Defines the 'radius' of the bounding box within which convolution takes place (also known as half-support).
 *                              Box width in pixels = 2*support + 1. The central pixel is the one nearest to the UV co-ordinates.
 *                              Default = 3.
 * @param[in] kernel_exact (bool): If true, calculates exact kernel values for every UV-sample. Otherwise, oversampling is used.
 *                             Default = true.
 * @param[in] kernel_oversampling (int): Controls kernel-generation if 'kernel_exact == False'. Larger values give a finer-sampled set of pre-cached kernels.
 *                                   Default = 9.
 * @param[in] generate_beam (bool): Whether or not to compute the beam matrix.
 *                                  Default = false.
 * @param[in] r_fft (FFTRoutine): Selects FFT routine to be used.
 * @param[in] fft_wisdom_filename (string): FFTW wisdom filename for the image and beam (c2r fft).
 *
 * @return (pybind11::tuple): Tuple of numpy.ndarrays representing the image map and beam model (image, beam).
 */
pybind11::tuple image_visibilities_wrapper(
    np_complex_double_array vis, // numpy.ndarray<np.complex_>
    np_double_array snr_weights, // numpy.ndarray<np.float_>
    np_double_array uvw_lambda, // numpy.ndarray<np.float_>
    int image_size,
    double cell_size,
    stp::KernelFunction kernel_func, // enum
    double kernel_trunc_radius,
    int kernel_support,
    bool kernel_exact,
    int kernel_oversampling,
    bool generate_beam,
    stp::FFTRoutine r_fft, // enum
    std::string fft_wisdom_filename);

/**
 * @brief Convenience wrapper over source_find_image function.
 *
 * This function shall be called from python code.
 *
 * @param[in] image_data (numpy.ndarray<np.float_>): Real component of iFFT'd image data (typically uses image returned by image_visibilities function).
 * @param[in] detection_n_sigma (double): Detection threshold as multiple of RMS.
 * @param[in] analysis_n_sigma (double): Analysis threshold as multiple of RMS.
 * @param[in] rms_est (double): RMS estimate (may be 0.0, in which case RMS is estimated from the image data).
 *                          Default = 0.0.
 * @param[in] find_negative_sources (bool): Find also negative sources (with signal is -1). Default = true.
 * @param[in] sigmaclip_iters (uint): Number of iterations of sigma clip function. Default = 5.
 * @param[in] binapprox_median (bool): Compute approximated median using the fast binapprox method. Default = false.
 * @param[in] compute_barycentre (bool): Compute barycentric centre of each island. Default = true.
 * @param[in] gaussian_fitting (bool): Perform gaussian fitting for each island. Default = false.
 * @param[in] generate_labelmap (bool): Update the final label map by removing the sources below the detection threshold. Default = true.
 *
 * @return (pybind11::list): List of tuples representing the source-detections.
 *                           Tuple components are as follows: (sign, val, x_idx, y_idx, xbar, ybar,
 *                           amplitude, x0, y0, x_stddev, y_stddev, theta, ceres_report), where:
 *                             - 'sign' is +1 or -1 (int), representing whether the source is positive or negative;
 *                             - 'val' (double) is the 'extremum_val', i.e. max or min pixel value for the positive or negative source case;
 *                             - 'x_idx,y_idx' (int) are the pixel-index of the extremum value;
 *                             - 'xbar, ybar' (double) are 'centre-of-mass' locations for the source-detection island.
 * The following are invalid if gaussian fitting is false:
 *                             - Gaussian parameters (double): 'amplitude', 'x0', 'y0', 'x_stddev', 'y_stddev', 'theta'.
 *                             - Ceres solver brief report (string).
 */
std::vector<std::tuple<int, double, int, int, double, double, double, double, double, double, double, double, std::string>> source_find_wrapper(
    np_real_array image_data, // numpy.ndarray<np.float_>
    double detection_n_sigma,
    double analysis_n_sigma,
    double rms_est,
    bool find_negative_sources,
    uint sigma_clip_iters,
    bool binapprox_median,
    bool compute_barycentre,
    bool gaussian_fitting,
    bool generate_labelmap);
}

#endif /* STP_PYTHON_H */
