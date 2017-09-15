/**
* @file run sourcefind.cpp
* @brief Main file for run source find function
*
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
    arma::Mat<real_t> image = arma::conv_to<arma::Mat<real_t>>::from(arma::real(load_npy_complex_array(_inNpzFileArg.getValue(), "image")));

    // Load all configurations from json configuration file
    ConfigurationFile cfg(_inJsonFileArg.getValue());

    if (use_logger) {
        _logger->info("Configuration parameters:");
        _logger->info(" - detection_n_sigma={}", cfg.detection_n_sigma);
        _logger->info(" - analysis_n_sigma={}", cfg.analysis_n_sigma);
        _logger->info(" - rms_estimation={}", cfg.estimate_rms);
        _logger->info(" - sigma_clip_iters={}", cfg.sigma_clip_iters);
        _logger->info(" - binapprox_median={}", cfg.binapprox_median);
        _logger->info(" - gaussian_fitting={}", cfg.gaussian_fitting);
        _logger->info(" - generate_labelmap={}", cfg.generate_labelmap);
        _logger->info(" - ceres_diffmethod={}", cfg.s_ceres_diffmethod);
        _logger->info(" - ceres_solvertype={}", cfg.s_ceres_solvertype);
        _logger->info("Running source find");
    }

    // Run source find
    stp::SourceFindImage sfimage(std::move(image), cfg.detection_n_sigma, cfg.analysis_n_sigma, cfg.estimate_rms, true,
        cfg.sigma_clip_iters, cfg.binapprox_median, cfg.gaussian_fitting, cfg.generate_labelmap,
        cfg.ceres_diffmethod, cfg.ceres_solvertype);

    if (use_logger) {
        _logger->info("Finished pipeline execution");
        if (_outJsonFileArg.isSet() || _outNpzFileArg.isSet()) {
            _logger->info("Saving output data");
        }
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
            _logger->info(" * Island {}: label={}, sign={}, extremum_val={}, extremum_x_idx={}, extremum_y_idy={}",
                island_num, i.label_idx, i.sign, i.extremum_val, i.extremum_x_idx, i.extremum_y_idx);
            _logger->info("   Moments fit: amplitude={}, x_centre={}, y_centre={}, semimajor={}, semiminor={}, theta={}",
                i.moments_fit.amplitude, i.moments_fit.x_centre, i.moments_fit.y_centre, i.moments_fit.semimajor, i.moments_fit.semiminor, i.moments_fit.theta);
            if (cfg.gaussian_fitting) {
                _logger->info("   Bounding box: top={}, bottom={}, left={}, right={}, width={}, height={}",
                    i.bounding_box.top, i.bounding_box.bottom, i.bounding_box.left, i.bounding_box.right, i.bounding_box.get_width(), i.bounding_box.get_height());
                _logger->info("   Gaussian fit: amplitude={}, x_centre={}, y_centre={}, semimajor={}, semiminor={}, theta={}",
                    i.leastsq_fit.amplitude, i.leastsq_fit.x_centre, i.leastsq_fit.y_centre, i.leastsq_fit.semimajor, i.leastsq_fit.semiminor, i.leastsq_fit.theta);
                _logger->info("   {}", i.ceres_report);
                _logger->info("");
            }
            island_num++;
        }
    }

    return 0;
}
