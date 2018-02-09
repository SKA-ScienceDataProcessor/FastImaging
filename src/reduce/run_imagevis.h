/** @file run_imagevis.h
 *
 *  @brief Simulated run of image visibilities
 *
 */

#ifndef RUN_IMAGEVIS_H
#define RUN_IMAGEVIS_H

#include <armadillo>

// Spdlog
#include <spdlog/spdlog.h>

// TCLAP
#include <tclap/CmdLine.h>

// Logger
std::shared_ptr<spdlog::logger> _logger;

// Command line parser
TCLAP::CmdLine _cmd("Simulated run of image visibilities function", ' ', "0.1");
// Input Json config filename
TCLAP::UnlabeledValueArg<std::string> _inJsonFileArg("input-file-json", "Input JSON filename with configuration parameters.", true, "", "input-file-json");
// Input Npz filename
TCLAP::UnlabeledValueArg<std::string> _inNpzFileArg("input-file-npz", "Input NPZ filename with simulation data (uvw_lambda, vis, skymodel).", true, "", "input-file-npz");
// Output Npz filename
TCLAP::UnlabeledValueArg<std::string> _outNpzFileArg("output-file-npz", "(optional)  Output NPZ filename for image and beam matrices (image, beam).", false, "", "output-file-npz");
// Use residual visibilities - difference between 'input_vis' and 'model' visibilities
TCLAP::SwitchArg _useDiffArg("d", "diff", "Use residual visibilities - difference between 'input_vis' and 'model' visibilities. Input NPZ must contain 'skymodel' data.", false);
// Enable logger
TCLAP::SwitchArg _enableLoggerArg("l", "log", "Enable logger (write to standard output and file).", false);

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

#endif /* RUN_IMAGEVIS_H */
