/**
* @file reduce.cpp
* Main file of reduce
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
        _logger->info(" - kernel_function={}", cfg.s_kernel_function);
        _logger->info(" - fft_routine={}", cfg.s_fft_routine);
        _logger->info(" - image_fft_wisdom={}", cfg.image_wisdom_filename);
        _logger->info(" - beam_fft_wisdom={}", cfg.beam_wisdom_filename);
        _logger->info(" - rms_estimation={}", cfg.estimate_rms);
        _logger->info(" - sigma_clip_iters={}", cfg.sigma_clip_iters);
        _logger->info(" - binapprox_median={}", cfg.binapprox_median);
        _logger->info(" - compute_barycentre={}", cfg.compute_barycentre);
        _logger->info(" - generate_labelmap={}", cfg.generate_labelmap);
        _logger->info(" - normalize_beam={}", cfg.normalize_beam);
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
        result = stp::image_visibilities(kernel_function, std::move(input_vis), std::move(input_uvw),
            cfg.image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling, true,
            cfg.normalize_beam, cfg.fft_routine, cfg.image_wisdom_filename, cfg.beam_wisdom_filename);
    };
        break;
    case stp::KernelFunction::Triangle: {
        stp::Triangle kernel_function(cfg.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(input_vis), std::move(input_uvw),
            cfg.image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling, true,
            cfg.normalize_beam, cfg.fft_routine, cfg.image_wisdom_filename, cfg.beam_wisdom_filename);
    };
        break;
    case stp::KernelFunction::Sinc: {
        stp::Sinc kernel_function(cfg.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(input_vis), std::move(input_uvw),
            cfg.image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling, true,
            cfg.normalize_beam, cfg.fft_routine, cfg.image_wisdom_filename, cfg.beam_wisdom_filename);
    };
        break;
    case stp::KernelFunction::Gaussian: {
        stp::Gaussian kernel_function(cfg.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(input_vis), std::move(input_uvw),
            cfg.image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling, true,
            cfg.normalize_beam, cfg.fft_routine, cfg.image_wisdom_filename, cfg.beam_wisdom_filename);
    };
        break;
    case stp::KernelFunction::GaussianSinc: {
        stp::GaussianSinc kernel_function(cfg.kernel_support);
        result = stp::image_visibilities(kernel_function, std::move(input_vis), std::move(input_uvw),
            cfg.image_size, cfg.cell_size, cfg.kernel_support, cfg.kernel_exact, cfg.oversampling, true,
            cfg.normalize_beam, cfg.fft_routine, cfg.image_wisdom_filename, cfg.beam_wisdom_filename);
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
    stp::source_find_image sfimage = stp::source_find_image(std::move(result.first), cfg.detection_n_sigma, cfg.analysis_n_sigma,
        cfg.estimate_rms, true, cfg.sigma_clip_iters, cfg.binapprox_median, cfg.compute_barycentre, cfg.generate_labelmap);

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
            _logger->info(" * Island {}: label={}, sign={}, extremum_val={}, extremum_x_idx={}, extremum_y_idy={}, xbar={}, ybar={}",
                island_num, i.label_idx, i.sign, i.extremum_val, i.extremum_x_idx, i.extremum_y_idx, i.xbar, i.ybar);
            island_num++;
        }
    }

#ifdef FUNCTION_TIMINGS
    times_red.push_back(std::chrono::high_resolution_clock::now());

    // Display benchmarking times
    std::chrono::duration<double> time_span;
    _logger->info("Running time of each pipeline step:");

    std::vector<std::string> imager_steps = { "Gridder", "IFFT", "Normalise" };
    _logger->info(" Imager:");
    for (uint i = 1; i < stp::times_iv.size(); i++) {
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_iv[i] - stp::times_iv[i - 1]);
        _logger->info(" - {:11s} = {:10.5f}", imager_steps[i - 1], time_span.count());
    }
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_iv.back() - stp::times_iv.front());
    _logger->info(" - Total       = {:10.5f}", time_span.count());

    std::vector<std::string> sourcefind_steps = { "Bg_level", "RMS est", "Label det", "Islands" };
    _logger->info(" Source find:");
    for (uint i = 1; i < stp::times_sf.size(); i++) {
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_sf[i] - stp::times_sf[i - 1]);
        _logger->info(" - {:11s} = {:10.5f}", sourcefind_steps[i - 1], time_span.count());
    }
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_sf.back() - stp::times_sf.front());
    _logger->info(" - Total       = {:10.5f}", time_span.count());

    std::vector<std::string> reduce_steps = { "Read data", "Image vis", "Source find", "Write data" };
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
