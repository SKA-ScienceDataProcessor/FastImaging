/**
* @file run_imagevis.cpp
* Main file for run image visibilities function
* Contains the main function. Creates and configures the TCLAP interface.
* Calls image visibilities funtion and savess the results.
*/

#include "run_imagevis.h"

// STP library
#include <stp.h>

// Load NPZ simulation data
#include <load_data.h>

// Load JSON configuration
#include <load_json_config.h>

// Save NPZ output file
#include <save_data.h>

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
    _cmd.add(_useDiffArg);
    _cmd.add(_inJsonFileArg);
    _cmd.add(_inNpzFileArg);
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

    // Check if use residual visibilities
    bool use_residual = false;
    if (_useDiffArg.isSet()) {
        use_residual = true;
    }

    if (use_logger) {
        // Creates and initializes the logger
        initLogger();
        _logger->info("Loading data");
    }

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(_inNpzFileArg.getValue(), "uvw_lambda");
    arma::cx_mat input_vis = load_npy_complex_array(_inNpzFileArg.getValue(), "vis");
    if (use_residual) {
        arma::cx_mat input_model = load_npy_complex_array(_inNpzFileArg.getValue(), "model");
        // Subtract model-generated visibilities from incoming data
        input_vis -= input_model;
        _logger->info("Use residual visibilities - input visibilities subtracted from model visibilities");
    }

    // Load all configurations from json configuration file
    ConfigurationFile cfg(_inJsonFileArg.getValue());

    if (use_logger) {
        _logger->info("Configuration parameters:");
        _logger->info(" - image_size={}", cfg.image_size);
        _logger->info(" - cell_size={}", cfg.cell_size);
        _logger->info(" - kernel_support={}", cfg.kernel_support);
        _logger->info(" - kernel_exact={}", cfg.kernel_exact);
        _logger->info(" - oversampling={}", cfg.oversampling);
        _logger->info("Running image visibilities");
    }

    // Run image_visibilities
    std::pair<arma::cx_mat, arma::cx_mat> result = stp::image_visibilities(stp::GaussianSinc(cfg.kernel_support), input_vis, input_uvw, cfg.image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling);

    if (use_logger) {
        _logger->info("Finished");
        _logger->info("Saving output data");
    }

    // Save image and beam matrices in NPZ file
    if (_outNpzFileArg.isSet()) {
        npz_save(_outNpzFileArg.getValue(), "image", result.first, "w");
        npz_save(_outNpzFileArg.getValue(), "beam", result.second, "a");
    }

    return 0;
}
