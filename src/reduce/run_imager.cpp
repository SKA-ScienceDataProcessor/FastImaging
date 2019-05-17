/**
* @file run_imager.cpp
* @brief Main file for run imager function
*
* Contains the main function. Creates and configures the TCLAP interface.
* Calls image visibilities funtion and saves the results.
*/

#include "run_imager.h"
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

#ifdef FUNCTION_TIMINGS
std::vector<std::chrono::high_resolution_clock::time_point> times_main;
#define NUM_TIME_INST 10
#endif

// Logger global
std::shared_ptr<spdlog::logger> reducelogger;
std::shared_ptr<spdlog::logger> srclogger;
std::shared_ptr<spdlog::logger> benchlogger;

// Command line parser
TCLAP::CmdLine cmd("Simulated run of imager function", ' ', "2.0-beta");

// Input Json config filename
TCLAP::UnlabeledValueArg<std::string> inJsonFileArg("input-file-json", "Input JSON filename with configuration parameters.", true, "", "input-file-json");
// Input Npz filenames
TCLAP::UnlabeledValueArg<std::string> inNpzFileArg("input-file-npz", "Input NPZ filename with simulation data (uvw_lambda, vis, skymodel).", true, "", "input-file-npz");
// Output Json config filename
TCLAP::ValueArg<std::string> outJsonFileArg("s", "output-file-json", "(optional)  Output JSON filename for detected islands.", false, "", "output-file-json");
// Output Npz filenames
TCLAP::ValueArg<std::string> outNpzFileArg("o", "output-file-npz", "(optional)  Output NPZ filename for image and beam matrices (image, beam).", false, "", "output-file-npz");
// Use residual visibilities - difference between 'input_vis' and 'model' visibilities
TCLAP::SwitchArg useDiffArg("d", "diff", "Use residual visibilities - difference between 'input_vis' and 'model' visibilities. Input NPZ must contain 'skymodel' data.", false);
// Enable logger
TCLAP::MultiSwitchArg enableLoggerArg("l", "log", "Enable logger to stdout and logfile.txt (-ll for further debug logging of stp library).", false);
// Print benchmarks
TCLAP::SwitchArg disableBenchPrintArg("b", "no-bench", "Disable logging of function timings.", false);

void createFlags()
{
#ifdef FUNCTION_TIMINGS
    cmd.add(disableBenchPrintArg);
#endif
    cmd.add(enableLoggerArg);
    cmd.add(useDiffArg);
    cmd.add(inJsonFileArg);
    cmd.add(inNpzFileArg);
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

    // Load all configurations from json configuration file
    ConfigurationFile cfg(inJsonFileArg.getValue());

    // Set padded image size
    set_image_sizes(cfg.img_pars);

    // Log configuration
    log_configuration_imager(cfg);

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array<double>(inNpzFileArg.getValue(), "uvw_lambda");
    arma::cx_mat input_vis = load_npy_complex_array<double>(inNpzFileArg.getValue(), "vis");
    arma::mat input_snr_weights = load_npy_double_array<double>(inNpzFileArg.getValue(), "snr_weights");

#ifdef APROJECTION
    if (cfg.a_proj.isEnabled()) {
        cfg.a_proj.lha = load_npy_double_array<double>(inNpzFileArg.getValue(), "lha");
    }
#endif

    if (useDiffArg.isSet()) {
        // Load skymodel data
        arma::mat skymodel = load_npy_double_array<double>(inNpzFileArg.getValue(), "skymodel");
        // Generate model visibilities from the skymodel and UVW-baselines
        arma::cx_mat input_model = stp::generate_visibilities_from_local_skymodel(skymodel, input_uvw);

        // Subtract model-generated visibilities from incoming data
        input_vis -= input_model;
        reducelogger->info("Use residual visibilities - input visibilities subtracted from model visibilities");
    }

    reducelogger->info("Running image visibilities");

    TIMESTAMP_MAIN

    // Run imager
    stp::ImageVisibilities imager(std::move(input_vis), std::move(input_snr_weights),
        std::move(input_uvw), cfg.img_pars, cfg.w_proj, cfg.a_proj);

    TIMESTAMP_MAIN

    reducelogger->info("Finished pipeline execution");

    // Save image and beam matrices in NPZ file
    if (outNpzFileArg.isSet()) {
        reducelogger->info("Saving output data");
        npz_save(outNpzFileArg.getValue(), "image", imager.vis_grid, "w");
        npz_save(outNpzFileArg.getValue(), "beam", imager.sampling_grid, "a");
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
