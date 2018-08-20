/**
* @file run sourcefind.cpp
* @brief Main file for run source find function
*
* Contains the main function. Creates and configures the TCLAP interface.
* Calls source find funtion and saves the results.
*/

#include "run_sourcefind.h"
#include "common_functions.h"

// STP library
#include <stp.h>

// TCLAP
#include <tclap/CmdLine.h>

// Load NPZ simulation data
#include <load_data.h>

// Load JSON configuration
#include <load_json_config.h>

// Save NPZ output file
#include <save_data.h>

// Save JSON for source find output
#include <save_json_sf_output.h>

#ifdef FUNCTION_TIMINGS
std::vector<std::chrono::high_resolution_clock::time_point> times_main;
#define NUM_TIME_INST 10
#endif

// Logger global
std::shared_ptr<spdlog::logger> reducelogger;
std::shared_ptr<spdlog::logger> srclogger;
std::shared_ptr<spdlog::logger> benchlogger;

// Command line parser
TCLAP::CmdLine cmd("Simulated run of source find function", ' ', "2.0-beta");

// Input Json config filename
TCLAP::UnlabeledValueArg<std::string> inJsonFileArg("input-file-json", "Input JSON filename with configuration parameters.", true, "", "input-file-json");
// Input Npz filenames
TCLAP::UnlabeledValueArg<std::string> inNpzFileArg("input-file-npz", "Input NPZ filename with simulation data (image).", true, "", "input-file-npz");
// Output Json config filename
TCLAP::ValueArg<std::string> outJsonFileArg("s", "output-file-json", "(optional)  Output JSON filename for detected islands.", false, "", "output-file-json");
// Output Npz filenames
TCLAP::ValueArg<std::string> outNpzFileArg("o", "output-file-npz", "(optional)  Output NPZ filename for label map matrix (label_map).", false, "", "output-file-npz");
// Enable logger
TCLAP::MultiSwitchArg enableLoggerArg("l", "log", "Enable logger to stdout and logfile.txt (-ll for further debug logging of stp library).", false);
// Print islands
TCLAP::SwitchArg disableIslandPrintArg("n", "no-src", "Disable logging of detected islands.", false);
// Print benchmarks
TCLAP::SwitchArg disableBenchPrintArg("b", "no-bench", "Disable logging of function timings.", false);

void createFlags()
{
#ifdef FUNCTION_TIMINGS
    cmd.add(disableBenchPrintArg);
#endif
    cmd.add(disableIslandPrintArg);
    cmd.add(enableLoggerArg);
    cmd.add(inJsonFileArg);
    cmd.add(inNpzFileArg);
    cmd.add(outJsonFileArg);
    cmd.add(outNpzFileArg);
}

int main(int argc, char** argv)
{
#ifdef FUNCTION_TIMINGS
    times_main.reserve(NUM_TIME_INST);
#endif
    TIMESTAMP_MAIN

    // Adds the flags to the parser
    createFlags();

    // Parses the arguments from the command console
    cmd.parse(argc, argv);

    // Creates and initializes the logger
    initLogger(enableLoggerArg.getValue());

    reducelogger->info("Loading data");

    //Load simulated data from input npz file
    arma::Mat<real_t> image = load_npy_double_array<real_t>(inNpzFileArg.getValue(), "image");

    // Load all configurations from json configuration file
    ConfigurationFile cfg(inJsonFileArg.getValue());

    // Log configuration
    log_configuration_sourcefind(cfg);

    reducelogger->info("Running source find");

    TIMESTAMP_MAIN

    // Run source find
    stp::SourceFindImage sfimage(std::move(image), cfg.detection_n_sigma, cfg.analysis_n_sigma, cfg.estimate_rms,
        cfg.find_negative_sources, cfg.sigma_clip_iters, cfg.median_method, cfg.gaussian_fitting, cfg.ccl_4connectivity,
        cfg.source_min_area, cfg.generate_labelmap, cfg.ceres_diffmethod, cfg.ceres_solvertype);

    TIMESTAMP_MAIN

    reducelogger->info("Finished pipeline execution");
    if (outJsonFileArg.isSet() || outNpzFileArg.isSet()) {
        reducelogger->info("Saving output data");
    }

    // Save detected island parameters in JSON file
    if (outJsonFileArg.isSet()) {
        save_json_sourcefind_output(outJsonFileArg.getValue(), sfimage);
    }

    // Save label_map matrix in NPZ file
    if (outNpzFileArg.isSet()) {
        npz_save(outNpzFileArg.getValue(), "label_map", sfimage.label_map, "w");
    }

    // Output island parameters if logger is enabled
    reducelogger->info("Number of detected sources: {} ", sfimage.islands.size());
    if (!disableIslandPrintArg.isSet()) {
        log_detected_islands(sfimage, cfg.gaussian_fitting);
    }

#ifdef FUNCTION_TIMINGS
    // Display benchmarking times
    if (!disableBenchPrintArg.isSet()) {
        log_function_timings();
    }
#endif

    TIMESTAMP_MAIN

    return 0;
}
