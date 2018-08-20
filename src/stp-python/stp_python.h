/**
* @file stp_python.h
* @brief Prototypes of stp python wrapper functions
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
 *                                                   2D array, shape: (n_vis, 3). Assumed ordering is u,v,w.
 * @param[in] image_size (int): Width of the image in pixels. Assumes (image_size/2, image_size/2) corresponds to the origin in UV-space.
 *                              Must be multiple of 4.
 * @param[in] cell_size (double): Angular-width of a synthesized pixel in the image to be created (arcsecond).
 * @param[in] kernel_func (KernelFunction): Choice of kernel function from limited selection (see KernelFunction enum structure).
 *                                          Default = KernelFunction::GaussianSinc.
 * @param[in] kernel_support (int): Defines the 'radius' of the bounding box within which convolution takes place (also known as half-support).
 *                                  Box width in pixels = 2*support + 1. The central pixel is the one nearest to the UV co-ordinates.
 *                                  Default = 3.
 * @param[in] kernel_exact (bool): If true, calculates exact kernel values for every UV-sample. Otherwise, oversampling is used.
 *                                 Default = true.
 * @param[in] oversampling (int): Controls kernel-generation if 'kernel_exact == False'. Larger values give a finer-sampled set of pre-cached kernels.
 *                                Default = 8.
 * @param[in] generate_beam (bool): Whether or not to compute the beam matrix.
 *                                  Default = false.
 * @param[in] gridding_correction (bool): Corrects the gridding effect of the anti-aliasing kernel on the dirty image and beam model. Default is true.
 * @param[in] analytic_gcf (bool): Compute approximation of image-domain kernel from analytic expression of DFT. Default is false.
 * @param[in] r_fft (FFTRoutine): Selects FFT routine to be used.
 * @param[in] fft_wisdom_filename (string): Wisdom filename used by FFTW.
 * @param[in] num_wplanes (int): Number of planes for W-Projection. Set zero to disable W-projection. Default is 0.
 * @param[in] wplanes_median (bool): Use median to compute w-planes, otherwise use mean. Default is false.
 * @param[in] max_wpconv_support (int): Defines the maximum 'radius' of the bounding box within which convolution takes place when W-Projection is used.
 *                                      Box width in pixels = 2*support+1. Default is 0.
 * @param[in] hankel_opt (bool): Use Hankel Transform (HT) optimization for quicker execution of W-Projection. Set 0 to disable HT and 1 or 2 to enable HT.
 *                              The larger non-zero value increases HT accuracy, by using an extended W-kernel workarea size. Default is 0.
 * @param[in] undersampling_opt (int): Use W-kernel undersampling for faster kernel generation. Set 0 to disable undersampling and 1 to enable maximum
 *                                     undersampling. Reduce the level of undersampling by increasing the integer value. Default is 1.
 * @param[in] kernel_trunc_perc (float): Percentual value from maximum value at which w-kernel can be considered = 0. Default is 0 as in (0%).
 * @param[in] interp_type (std::string): Interpolation type to be used in the interpolation step in the Hankel Transorm.
 *                                     Available options are: "linear", "cosine" and "cubic".Default = "linear".
 * @param[in] aproj_numtimesteps (int): Number of time steps used for A-projection. Set zero to disable A-projection. Default is 0.
 * @param[in] obs_dec (double): Declination of observation pointing centre (in degrees). Default is 0.
 * @param[in] obs_lat (double): Latitude of observation pointing centre (in degrees). Default is 0.
 * @param[in] lha (arma::mat): Local hour angle of visibilities. LHA=0 is transit, LHA=-6h is rising, LHA=+6h is setting.
 * @param[in] mueller_term (arma::mat): Mueller matrix term (defined each image coordinate) used for A-projection.
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
    int kernel_support,
    bool kernel_exact,
    int oversampling,
    bool generate_beam,
    bool gridding_correction,
    bool analytic_gcf,
    stp::FFTRoutine r_fft, // enum
    std::string fft_wisdom_filename,
    int num_wplanes,
    bool wplanes_median,
    int max_wpconv_support,
    bool hankel_opt,
    int undersampling_opt,
    double kernel_trunc_perc,
    stp::InterpType interp_type, // enum
    int aproj_numtimesteps,
    double obs_dec,
    double obs_lat,
    np_double_array lha, // numpy.ndarray<np.float_>
    np_double_array mueller_term); // numpy.ndarray<np.float_>

/**
 * @brief Convenience wrapper over SourceFindImage function.
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
 * @param[in] median_method (MedianMethod): Method used to compute the median. Default = BINAPPROX.
 * @param[in] gaussian_fitting (bool): Perform gaussian fitting for each island. Default = false.
 * @param[in] ccl_4connectivity (bool): Use 4-connected component labeling for source find (default is 8-connected component labeling).
 * @param[in] generate_labelmap (bool): Update the final label map by removing the sources below the detection threshold. Default = true.
 * @param[in] source_min_area (int): Minimum number of pixels required for a source. Default is 5.
 * @param[in] ceres_diffmethod (CeresDiffMethod): Differentiation method used by ceres library for gaussian fitting.
 * @param[in] ceres_solvertype (CeresSolverType): Solver type used by ceres library for gaussian fitting.
 *
 * @return (pybind11::list): List of tuples representing the source-detections.
 *                           Tuple components are as follows: (sign, val, x_idx, y_idx, xbar, ybar, gaussian_fit ceres_log), where:
 *                             - 'sign' is +1 or -1 (int), representing whether the source is positive or negative;
 *                             - 'val' (double) is the 'extremum_val', i.e. max or min pixel value for the positive or negative source case;
 *                             - 'x_idx,y_idx' (int) are the pixel-index of the extremum value;
 *                             - 'num_samples' (int) number of samples in the island;
 *                             - 'moments_fit' (Gaussian2dParams) represents initial 2D gaussian fitting estimated from moments method:
 *                                                             'amplitude', 'x_centre', 'y_centre', 'x_stddev', 'y_stddev', 'theta'.
 * The following are valid only if input gaussian_fitting flag is enabled:
 *                             - 'gaussian_fit' (Gaussian2dParams) represents 2D gaussian fitting estimated from least-squares method:
 *                                                              'amplitude', 'x_centre', 'y_centre', 'x_stddev', 'y_stddev', 'theta'.
 *                             - 'ceres_report' (string) is the ceres solver report.
 */
std::vector<std::tuple<int, double, int, int, int, stp::Gaussian2dParams, stp::Gaussian2dParams, std::string>> source_find_wrapper(
    np_real_array image_data, // numpy.ndarray<np.float_>
    double detection_n_sigma,
    double analysis_n_sigma,
    double rms_est,
    bool find_negative_sources,
    uint sigma_clip_iters,
    stp::MedianMethod median_method,
    bool gaussian_fitting,
    bool ccl_4connectivity,
    bool generate_labelmap,
    int source_min_area,
    stp::CeresDiffMethod ceres_diffmethod,
    stp::CeresSolverType ceres_solvertype);
}

#endif /* STP_PYTHON_H */
