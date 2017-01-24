/** @file reduce.h
 *
 *  @brief Simulated pipeline run
 *
 */

#ifndef REDUCE_H
#define REDUCE_H

#include <armadillo>

// STP library
#include <stp.h>

// Load npz data
#include <load_data.h>

// Load JSON config
#include <load_json_config.h>

// Spdlog
#include <spdlog/spdlog.h>

// TCLAP
#include <tclap/CmdLine.h>

// Logger
std::shared_ptr<spdlog::logger> _logger;

// Command line parser
TCLAP::CmdLine _cmd("Simulated pipeline run for STP", ' ', "0.1");
// Input npz filename flag
TCLAP::ValueArg<std::string> _fileArg("f", "input_npz", "Input NPZ filename", true, "simdata.npz", "input_npz");
// Config json filename flag
TCLAP::ValueArg<std::string> _confArg("c", "json_config", "Json configuration filename", true, "fastimg_config.json", "json_config");

/**
* @brief Logger initialization function
*
* Creates and initializes the logger to be used throughout the program
*/
void initLogger();

/**
* @brief Creates the switch flags to be used by the parser
*
* Creates a list of switch arguments and then xor adds to the command line parser, meaning that only
* one flag and one only is required to run the program_invocation_name
*
*/
void createFlags();

/**
 * @brief Simulated pipeline run
 *
 * Search for transients as follows:
 *  - Apply difference imaging (subtract model visibilities from data, apply synthesis-imaging).
 *  - Run sourcefinding on the resulting diff-image.
 *
 * @param[in] uvw_lambda (mat) : UVW-coordinates of visibilities. Units are multiples of wavelength.
 * @param[in] model_vis (cx_mat): Input model visibilities.
 * @param[in] data_vis (cx_mat): Input complex visibilities.
 * @param[in] image_size (int): Width of the image in pixels
 * @param[in] cell_size (double): Angular-width of a synthesized pixel in the image to be created
 * @param[in] kernel_support (int): Defines the 'radius' of the bounding box within which convolution takes place
 * @param[in] kernel_exact (bool): Calculate exact kernel-values for every UV-sample
 * @param[in] oversampling (int): Controls kernel-generation if ``exact==False``. Larger values give a finer-sampled set of pre-cached kernels
 * @param[in] detection_n_sigma (double): Detection threshold as multiple of RMS
 * @param[in] analysis_n_sigma (double): Analysis threshold as multiple of RMS
 *
 * @return result of the pipeline
 */
stp::source_find_image run_pipeline(arma::mat uvw_lambda, arma::cx_mat model_vis, arma::cx_mat data_vis, int image_size, double cell_size, int kernel_support, bool kernel_exact, int oversampling, double detection_n_sigma, double analysis_n_sigma);

#endif /* REDUCE_H */
