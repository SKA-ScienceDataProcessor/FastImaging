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
TCLAP::UnlabeledValueArg<std::string> _inNpzFileArg("input-file-npz", "Input NPZ filename with simulation data (uvw_lambda, vis, skymodel).", true, "", "input-file-npz");
// Output Json config filename
TCLAP::UnlabeledValueArg<std::string> _outJsonFileArg("output-file-json", "Output JSON filename for detected islands.", true, "", "output-file-json");
// Output Npz filename
TCLAP::UnlabeledValueArg<std::string> _outNpzFileArg("output-file-npz", "(optional)  Output NPZ filename for label map matrix (label_map).", false, "", "output-file-npz");
// Use residual visibilities - difference between 'input_vis' and 'model' visibilities
TCLAP::SwitchArg _useDiffArg("d", "diff", "Use residual visibilities - difference between 'input_vis' and 'model' visibilities. Input NPZ must contain 'skymodel' data.", false);
// Enable logger
TCLAP::SwitchArg _enableLoggerArg("l", "log", "Enable logger (write to standard output and file).", false);
// Print islands
TCLAP::SwitchArg _enableIslandPrintArg("p", "print", "Print islands found to standard output.", false);

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

#endif /* REDUCE_H */
