/** @file run_sourcefind.h
 *
 *  @brief Simulated run of source find
 *
 */

#ifndef RUN_SOURCEFIND_H
#define RUN_SOURCEFIND_H

#include <armadillo>

// Spdlog
#include <spdlog/spdlog.h>

// TCLAP
#include <tclap/CmdLine.h>

// Logger
std::shared_ptr<spdlog::logger> _logger;

// Command line parser
TCLAP::CmdLine _cmd("Simulated run of source find function", ' ', "0.1");
// Input Json config filename
TCLAP::UnlabeledValueArg<std::string> _inJsonFileArg("input-file-json", "Input JSON filename with configuration parameters.", true, "", "input-file-json");
// Input Npz filename
TCLAP::UnlabeledValueArg<std::string> _inNpzFileArg("input-file-npz", "Input NPZ filename with simulation data (image).", true, "", "input-file-npz");
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

#endif /* RUN_SOURCEFIND_H */
