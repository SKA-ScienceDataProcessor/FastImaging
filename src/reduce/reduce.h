/** @file reduce.h
 *
 *  @brief Simulated slow transients pipeline run
 *
 */

#ifndef REDUCE_H
#define REDUCE_H

#include <armadillo>

// STP library
#include <stp.h>

// Spdlog
#include <spdlog/spdlog.h>

// TCLAP
#include <tclap/CmdLine.h>

// Logger
std::shared_ptr<spdlog::logger> _logger;

// Command line parser
TCLAP::CmdLine _cmd("Simulated Slow Transients Pipeline run", ' ', "0.1");
// Input Json config filename
TCLAP::UnlabeledValueArg<std::string> _inJsonFileArg("input-file-json", "Input JSON filename with configuration parameters.", true, "", "input-file-json");
// Input Npz filename
TCLAP::UnlabeledValueArg<std::string> _inNpzFileArg("input-file-npz", "Input NPZ filename with simulation data (uvw_lambda, model, vis).", true, "", "input-file-npz");
// Output Json config filename
TCLAP::UnlabeledValueArg<std::string> _outJsonFileArg("output-file-json", "Output JSON filename for detected islands.", true, "", "output-file-json");
// Output Npz filename
TCLAP::UnlabeledValueArg<std::string> _outNpzFileArg("output-file-npz", "(optional)  Output NPZ filename for label map matrix (label_map).", false, "", "output-file-npz");
// Enable logger
TCLAP::SwitchArg _enableLoggerArg("l", "log", "Enable logger.", false);

/**
* @brief Logger initialization function
*
* Creates and initializes the logger to be used throughout the program
*/
void initLogger();

/**
* @brief Creates the switch flags to be used by the parser
*
* Creates a list of switch arguments to the command line parser
*
*/
void createFlags();

/**
 * @brief Simulated slow trasients pipeline run
 *
 * Search for slow transients as follows:
 *  - Apply difference imaging (subtract model visibilities from data, apply synthesis-imaging).
 *  - Run sourcefinding on the resulting diff-image.
 *
 * @param[in] uvw_lambda (arma::mat) : UVW-coordinates of visibilities. Units are multiples of wavelength.
 * @param[in] residual_vis (arma::cx_mat): Input residual visibilities.
 * @param[in] image_size (int): Width of the image in pixels.
 * @param[in] cell_size (double): Angular-width of a synthesized pixel in the image to be created.
 * @param[in] kernel_support (int): Defines the 'radius' of the bounding box within which convolution takes place.
 * @param[in] kernel_exact (bool): Calculate exact kernel-values for every UV-sample.
 * @param[in] oversampling (int): Controls kernel-generation if ``exact==False``. Larger values give a finer-sampled set of pre-cached kernels.
 * @param[in] detection_n_sigma (double): Detection threshold as multiple of RMS.
 * @param[in] analysis_n_sigma (double): Analysis threshold as multiple of RMS.
 *
 * @return result of the pipeline
 */
stp::source_find_image run_pipeline(arma::mat& uvw_lambda, arma::cx_mat& residual_vis, int image_size, double cell_size, int kernel_support, bool kernel_exact, int oversampling, double detection_n_sigma, double analysis_n_sigma);

#endif /* REDUCE_H */
