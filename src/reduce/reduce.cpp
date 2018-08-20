/**
* @file reduce.cpp
* @brief Main file of reduce
*
* Contains the main function. Creates and configures the TCLAP interface.
* Calls pipeline funtion and saves the results.
*/

#include "reduce.h"
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
TCLAP::CmdLine cmd("Simulated Slow Transients Pipeline run", ' ', "2.0-beta");

// Input Json config filename
TCLAP::UnlabeledValueArg<std::string> inJsonFileArg("input-file-json", "Input JSON filename with configuration parameters.", true, "", "input-file-json");
// Input Npz filenames
TCLAP::UnlabeledValueArg<std::string> inNpzFileArg("input-file-npz", "Input NPZ filename with simulation data (uvw_lambda, vis, skymodel).", true, "", "input-file-npz");
// Output Json config filename
TCLAP::ValueArg<std::string> outJsonFileArg("s", "output-file-json", "(optional)  Output JSON filename for detected islands.", false, "", "output-file-json");
// Output Npz filenames
TCLAP::ValueArg<std::string> outNpzFileArg("o", "output-file-npz", "(optional)  Output NPZ filename for label map matrix (label_map).", false, "", "output-file-npz");
// Use residual visibilities - difference between 'input_vis' and 'model' visibilities
TCLAP::SwitchArg useDiffArg("d", "diff", "Use residual visibilities - difference between 'input_vis' and 'model' visibilities. Input NPZ must contain 'skymodel' data.", false);
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
    cmd.add(useDiffArg);
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

    // Load all configurations from json configuration file
    ConfigurationFile cfg(inJsonFileArg.getValue());

    // Image size must be a power of two value
    check_image_size(cfg.img_pars.image_size);

    // Log configuration
    log_configuration_imager(cfg);
    log_configuration_sourcefind(cfg);

    //Load simulated data (UVW-baselines and visibilities) from input_npz
    arma::mat input_uvw = load_npy_double_array<double>(inNpzFileArg.getValue(), "uvw_lambda");
    arma::cx_mat input_vis = load_npy_complex_array<double>(inNpzFileArg.getValue(), "vis");
    arma::mat input_snr_weights = load_npy_double_array<double>(inNpzFileArg.getValue(), "snr_weights");

#ifdef APROJECTION
    if (cfg.a_proj.isEnabled()) {
        cfg.a_proj.lha = load_npy_double_array<double>(inNpzFileArg.getValue(), "lha");
        cfg.a_proj.mueller_term = arma::ones(cfg.img_pars.image_size, cfg.img_pars.image_size);
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

    reducelogger->info("Running pipeline");

    TIMESTAMP_MAIN

    // Run imager
    stp::ImageVisibilities imager(std::move(input_vis), std::move(input_snr_weights),
        std::move(input_uvw), cfg.img_pars, cfg.w_proj, cfg.a_proj);

    imager.sampling_grid.reset(); // Destroy unused matrix

    TIMESTAMP_MAIN

    // Run source find
    stp::SourceFindImage sfimage(std::move(imager.vis_grid), cfg.detection_n_sigma, cfg.analysis_n_sigma, cfg.estimate_rms,
        cfg.find_negative_sources, cfg.sigma_clip_iters, cfg.median_method, cfg.gaussian_fitting, cfg.ccl_4connectivity,
        cfg.generate_labelmap, cfg.source_min_area, cfg.ceres_diffmethod, cfg.ceres_solvertype);

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
