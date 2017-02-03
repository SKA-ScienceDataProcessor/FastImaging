/**
* @file run sourcefind.cpp
* Main file for run source find function
* Contains the main function. Creates and configures the TCLAP interface.
* Calls source find funtion and saves the results.
*/

#include "run_sourcefind.h"

// STP library
#include <stp.h>

// Load NPZ simulation data
#include <load_data.h>

// Load JSON configuration
#include <load_json_config.h>

// Save NPZ output file
#include <save_data.h>

// Save JSON for source find output
#include <save_json_sf_output.h>

void initLogger()
{
    // Creates two spdlog sinks
    // One sink for the stdout and another for a file
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_st>());
    sinks.push_back(std::make_shared<spdlog::sinks::simple_file_sink_st>("logfile.txt"));
    _logger = std::make_shared<spdlog::logger>("logger", begin(sinks), end(sinks));
}

void createFlags()
{
    _cmd.add(_enableLoggerArg);
    _cmd.add(_inJsonFileArg);
    _cmd.add(_inNpzFileArg);
    _cmd.add(_outJsonFileArg);
    _cmd.add(_outNpzFileArg);
}

int main(int argc, char** argv)
{
    // Adds the flags to the parser
    createFlags();

    // Parses the arguments from the command console
    _cmd.parse(argc, argv);

    // Check if use logger
    bool use_logger = false;
    if (_enableLoggerArg.isSet()) {
        use_logger = true;
    }

    if (use_logger) {
        // Creates and initializes the logger
        initLogger();
        _logger->info("Loading data");
    }

    //Load simulated data from input npz file
    arma::mat image = arma::real(load_npy_complex_array(_inNpzFileArg.getValue(), "image"));

    // Load all configurations from json configuration file
    ConfigurationFile cfg(_inJsonFileArg.getValue());

    if (use_logger) {
        _logger->info("Configuration parameters:");
        _logger->info(" - detection_n_sigma={}", cfg.detection_n_sigma);
        _logger->info(" - analysis_n_sigma={}", cfg.analysis_n_sigma);
        _logger->info("Running source find");
    }

    // Run source find
    stp::source_find_image sfimage = stp::source_find_image(image, cfg.detection_n_sigma, cfg.analysis_n_sigma, 0.0, true);

    if (use_logger) {
        _logger->info("Finished");
        _logger->info("Saving output data");
    }

    // Save detected island parameters in JSON file
    if (_outJsonFileArg.isSet()) {
        save_json_sourcefind_output(_outJsonFileArg.getValue(), sfimage);
    }

    // Save label_map matrix in NPZ file
    if (_outNpzFileArg.isSet()) {
        npz_save(_outNpzFileArg.getValue(), "label_map", sfimage.label_map, "w");
    }

    // Output island parameters if logger is enabled
    if (use_logger) {
        _logger->info("Number of found islands: {} ", sfimage.islands.size());

        std::size_t island_num = 0;
        for (auto&& i : sfimage.islands) {
            _logger->info(" * Island {}: label={}, sign={}, extremum_val={}, extremum_x_idx={}, extremum_y_idy={}, xbar={}, ybar={}",
                island_num, i.label_idx, i.sign, i.extremum_val, i.extremum_x_idx, i.extremum_y_idx, i.xbar, i.ybar);
            island_num++;
        }
    }

    return 0;
}
