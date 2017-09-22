/**
* @file reduce.cpp
* @brief Main file of reduce
*
* Contains the main function. Creates and configures the TCLAP interface.
* Calls pipeline funtion and saves the results.
*/

#include "reduce.h"

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
    _cmd.add(_useDiffArg);
    _cmd.add(_inJsonFileArg);
    _cmd.add(_inNpzFileArg);
    _cmd.add(_outJsonFileArg);
    _cmd.add(_outNpzFileArg);
}

int main(int argc, char** argv)
{
#ifdef FUNCTION_TIMINGS
    std::vector<std::chrono::high_resolution_clock::time_point> times_red;
    times_red.reserve(NUM_TIME_INST);
    times_red.push_back(std::chrono::high_resolution_clock::now());
#endif

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

    //Load simulated data (UVW-baselines and visibilities) from input_npz
    arma::mat input_uvw = load_npy_double_array(_inNpzFileArg.getValue(), "uvw_lambda");
    arma::cx_mat input_vis = load_npy_complex_array(_inNpzFileArg.getValue(), "vis");
    arma::mat input_snr_weights = load_npy_double_array(_inNpzFileArg.getValue(), "snr_weights");

    if (use_residual) {
        // Load skymodel data
        arma::mat skymodel = load_npy_double_array(_inNpzFileArg.getValue(), "skymodel");
        // Generate model visibilities from the skymodel and UVW-baselines
        arma::cx_mat input_model = stp::generate_visibilities_from_local_skymodel(skymodel, input_uvw);

        // Subtract model-generated visibilities from incoming data
        input_vis -= input_model;
        if (use_logger) {
            _logger->info("Use residual visibilities - input visibilities subtracted from model visibilities");
        }
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
        _logger->info(" - detection_n_sigma={}", cfg.detection_n_sigma);
        _logger->info(" - analysis_n_sigma={}", cfg.analysis_n_sigma);
        _logger->info(" - find_negative_sources={}", cfg.find_negative_sources);
        _logger->info(" - kernel_function={}", cfg.s_kernel_function);
        _logger->info(" - fft_routine={}", cfg.s_fft_routine);
        _logger->info(" - fft_wisdom_file={}", cfg.fft_wisdom_filename);
        _logger->info(" - rms_estimation={}", cfg.estimate_rms);
        _logger->info(" - sigma_clip_iters={}", cfg.sigma_clip_iters);
        _logger->info(" - median_method={}", cfg.s_median_method);
        _logger->info(" - gaussian_fitting={}", cfg.gaussian_fitting);
        _logger->info(" - generate_labelmap={}", cfg.generate_labelmap);
        _logger->info(" - generate_beam={}", cfg.generate_beam);
        _logger->info(" - ceres_diffmethod={}", cfg.s_ceres_diffmethod);
        _logger->info(" - ceres_solvertype={}", cfg.s_ceres_solvertype);
        _logger->info("Running pipeline");
    }

    // Create output matrix
    std::pair<arma::Mat<real_t>, arma::Mat<real_t>> result;

#ifdef FUNCTION_TIMINGS
    times_red.push_back(std::chrono::high_resolution_clock::now());
#endif

    // Run image_visibilities
    switch (cfg.kernel_function) {
    case stp::KernelFunction::TopHat: {
        stp::TopHat kernel_function(cfg.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(input_vis), std::move(input_snr_weights),
            std::move(input_uvw), cfg.image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling,
            cfg.generate_beam, cfg.fft_routine, cfg.fft_wisdom_filename);
    };
        break;
    case stp::KernelFunction::Triangle: {
        stp::Triangle kernel_function(cfg.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(input_vis), std::move(input_snr_weights),
            std::move(input_uvw), cfg.image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling,
            cfg.generate_beam, cfg.fft_routine, cfg.fft_wisdom_filename);
    };
        break;
    case stp::KernelFunction::Sinc: {
        stp::Sinc kernel_function(cfg.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(input_vis), std::move(input_snr_weights),
            std::move(input_uvw), cfg.image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling,
            cfg.generate_beam, cfg.fft_routine, cfg.fft_wisdom_filename);
    };
        break;
    case stp::KernelFunction::Gaussian: {
        stp::Gaussian kernel_function(cfg.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(input_vis), std::move(input_snr_weights),
            std::move(input_uvw), cfg.image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling,
            cfg.generate_beam, cfg.fft_routine, cfg.fft_wisdom_filename);
    };
        break;
    case stp::KernelFunction::GaussianSinc: {
        stp::GaussianSinc kernel_function(cfg.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(input_vis), std::move(input_snr_weights),
            std::move(input_uvw), cfg.image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling,
            cfg.generate_beam, cfg.fft_routine, cfg.fft_wisdom_filename);
    };
        break;
    default:
        assert(0);
        break;
    }

    result.second.reset(); // Destroy unused matrix

#ifdef FUNCTION_TIMINGS
    times_red.push_back(std::chrono::high_resolution_clock::now());
#endif

    // Run source find
    stp::SourceFindImage sfimage(std::move(result.first), cfg.detection_n_sigma, cfg.analysis_n_sigma, cfg.estimate_rms,
        cfg.find_negative_sources, cfg.sigma_clip_iters, cfg.median_method, cfg.gaussian_fitting, cfg.generate_labelmap,
        cfg.ceres_diffmethod, cfg.ceres_solvertype);

#ifdef FUNCTION_TIMINGS
    times_red.push_back(std::chrono::high_resolution_clock::now());
#endif

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

#ifdef FUNCTION_TIMINGS
    times_red.push_back(std::chrono::high_resolution_clock::now());

    // Display benchmarking times
    std::chrono::duration<double> time_span;
    _logger->info("Running time of each pipeline step:");

    std::vector<std::string> imager_steps = { "Gridder", "FFT", "Normalise" };
    _logger->info(" Imager:");
    for (uint i = 1; i < stp::times_iv.size(); i++) {
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_iv[i] - stp::times_iv[i - 1]);
        _logger->info(" - {:11s} = {:10.5f}", imager_steps[i - 1], time_span.count());
    }
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_iv.back() - stp::times_iv.front());
    _logger->info(" - Total       = {:10.5f}", time_span.count());

    std::vector<std::string> sourcefind_steps = { "Bg level", "RMS est", "Labeling", "GaussianFit" };
    _logger->info(" Source find:");
    for (uint i = 1; i < stp::times_sf.size(); i++) {
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_sf[i] - stp::times_sf[i - 1]);
        _logger->info(" - {:11s} = {:10.5f}", sourcefind_steps[i - 1], time_span.count());
    }
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_sf.back() - stp::times_sf.front());
    _logger->info(" - Total       = {:10.5f}", time_span.count());

    std::vector<std::string> reduce_steps = { "Read data", "Imager", "Source find", "Write data" };
    _logger->info(" Reduce:");
    for (uint i = 1; i < times_red.size(); i++) {
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(times_red[i] - times_red[i - 1]);
        _logger->info(" - {:11s} = {:10.5f}", reduce_steps[i - 1], time_span.count());
    }
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(times_red.back() - times_red.front());
    _logger->info(" - Total       = {:10.5f}", time_span.count());
#endif

    return 0;
}
