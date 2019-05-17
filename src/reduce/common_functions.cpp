#include "common_functions.h"
// Spdlog
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <spdlog/async.h>
#include <spdlog/sinks/null_sink.h>

void initLogger(int logging_value)
{
    // Creates two spdlog sinks
    // One sink for the stdout and another for a file
    std::vector<spdlog::sink_ptr> reducesinks;
    // Create thread pool for async logger (spdlog does not provide factory function for loggers combining
    // multiple sinks, so we create async logger here - see spdlog README file)
    spdlog::init_thread_pool(8192, 1);
    auto stdout_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logfile.txt");
    if (logging_value) {
        reducesinks.push_back(stdout_sink);
        reducesinks.push_back(file_sink);
        if (logging_value > 1) {
            auto stpliblogger = std::make_shared<spdlog::async_logger>("stplib", begin(reducesinks), end(reducesinks), spdlog::thread_pool());
            stpliblogger->set_level(spdlog::level::debug);
            spdlog::register_logger(stpliblogger);
        }
    } else {
        reducesinks.push_back(std::make_shared<spdlog::sinks::null_sink_mt>());
    }

    // Logger used by executable
    reducelogger = std::make_shared<spdlog::async_logger>("reduce", begin(reducesinks), end(reducesinks), spdlog::thread_pool());
    reducelogger->set_level(spdlog::level::info);
    spdlog::register_logger(reducelogger);

    // Sinks for other loggers
    std::vector<spdlog::sink_ptr> sinks{ stdout_sink, file_sink };

    // Logger used to print list of sources
    srclogger = std::make_shared<spdlog::async_logger>("sources", begin(sinks), end(sinks), spdlog::thread_pool());
    srclogger->set_level(spdlog::level::info);
    spdlog::register_logger(srclogger);

    // Logger used to print benchmark timings
    benchlogger = std::make_shared<spdlog::async_logger>("benchmark", begin(sinks), end(sinks), spdlog::thread_pool());
    benchlogger->set_level(spdlog::level::info);
    spdlog::register_logger(benchlogger);
}

void set_image_sizes(stp::ImagerPars &imagerPars)
{
    // Image sizes must be multiple of 4
    bool changed = false;
    while ((imagerPars.image_size % 4) != 0) {
        imagerPars.image_size++;
        changed = true;
    }
    if (changed)
    {
        reducelogger->warn("!");
        reducelogger->warn("Image size must be a multiple of four.");
        reducelogger->warn("Set padded image size to {} pixels.", imagerPars.image_size);
        reducelogger->warn("!");
    }

    changed = false;
    imagerPars.padded_image_size = static_cast<double>(imagerPars.image_size) * imagerPars.padding_factor;
    while ((imagerPars.padded_image_size % 4) != 0) {
        imagerPars.padded_image_size++;
        changed = true;
    }
    if (changed)
    {
        reducelogger->warn("!");
        reducelogger->warn("Padded image size must be a multiple of four.");
        reducelogger->warn("Set padded image size to {} pixels.", imagerPars.padded_image_size);
        reducelogger->warn("!");
    }
}

void log_configuration_imager(const ConfigurationFile& cfg)
{
    reducelogger->info("Imager settings:");
    reducelogger->info(" - image_size={}", cfg.img_pars.image_size);
    reducelogger->info(" - cell_size={}", cfg.img_pars.cell_size);
    reducelogger->info(" - image_padding_factor={}", cfg.img_pars.padding_factor);
    reducelogger->info(" - padded_image_size={}", cfg.img_pars.padded_image_size);
    reducelogger->info(" - kernel_function={}", cfg.s_kernel_function);
    reducelogger->info(" - kernel_support={}", cfg.img_pars.kernel_support);
    reducelogger->info(" - kernel_exact={}", cfg.img_pars.kernel_exact);
    reducelogger->info(" - oversampling={}", cfg.img_pars.oversampling);
    reducelogger->info(" - generate_beam={}", cfg.img_pars.generate_beam);
    reducelogger->info(" - gridding_correction={}", cfg.img_pars.gridding_correction);
    reducelogger->info(" - analytic_gcf={}", cfg.img_pars.analytic_gcf);
    reducelogger->info(" - fft_routine={}", cfg.s_fft_routine);
    reducelogger->info(" - fft_wisdom_file={}", cfg.img_pars.fft_wisdom_filename);

#ifdef WPROJECTION
    if (cfg.w_proj.num_wplanes > 0) {
        reducelogger->info("W-Projection settings:");
        reducelogger->info(" - num_wplanes={}", cfg.w_proj.num_wplanes);
        reducelogger->info(" - wplanes_median={}", cfg.w_proj.wplanes_median);
        reducelogger->info(" - max_wpconv_support={}", cfg.w_proj.max_wpconv_support);
        reducelogger->info(" - hankel_opt={}", cfg.w_proj.hankel_opt);
        reducelogger->info(" - undersampling_opt={}", cfg.w_proj.undersampling_opt);
        reducelogger->info(" - kernel_trunc_perc={}", cfg.w_proj.kernel_trunc_perc);
        reducelogger->info(" - interp_type={}", cfg.s_interp_type);
    } else
#endif
        reducelogger->info("W-Projection disabled");

#ifdef APROJECTION
    if (cfg.a_proj.num_timesteps > 0) {
        reducelogger->info("A-Projection settings:");
        reducelogger->info(" - aproj_numtimesteps={}", cfg.a_proj.num_timesteps);
        reducelogger->info(" - obs_dec={}", cfg.a_proj.obs_dec);
        reducelogger->info(" - obs_ra={}", cfg.a_proj.obs_ra);
        reducelogger->info(" - aproj_opt={}", cfg.a_proj.aproj_opt);
        reducelogger->info(" - aproj_mask_perc={}", cfg.a_proj.aproj_mask_perc);
        reducelogger->info(" - pbeam_coefs={{{}}}", fmt::join(cfg.a_proj.pbeam_coefs.begin(), cfg.a_proj.pbeam_coefs.end(), ", "));
    } else
#endif
        reducelogger->info("A-Projection disabled");
}

void log_configuration_sourcefind(const ConfigurationFile& cfg)
{
    reducelogger->info("Source find settings:");
    reducelogger->info(" - detection_n_sigma={}", cfg.detection_n_sigma);
    reducelogger->info(" - analysis_n_sigma={}", cfg.analysis_n_sigma);
    reducelogger->info(" - find_negative_sources={}", cfg.find_negative_sources);
    reducelogger->info(" - rms_estimation={}", cfg.estimate_rms);
    reducelogger->info(" - sigma_clip_iters={}", cfg.sigma_clip_iters);
    reducelogger->info(" - median_method={}", cfg.s_median_method);
    reducelogger->info(" - gaussian_fitting={}", cfg.gaussian_fitting);
    reducelogger->info(" - ccl_4connectivity={}", cfg.ccl_4connectivity);
    reducelogger->info(" - generate_labelmap={}", cfg.generate_labelmap);
    reducelogger->info(" - source_min_area={}", cfg.source_min_area);
    reducelogger->info(" - ceres_diffmethod={}", cfg.s_ceres_diffmethod);
    reducelogger->info(" - ceres_solvertype={}", cfg.s_ceres_solvertype);
}

void log_detected_islands(stp::SourceFindImage& sfimage, bool print_gaussian_fit)
{
    srclogger->info("");
    srclogger->info("### LIST OF DETECTED SOURCES ###");
    std::size_t island_num = 0;
    for (auto&& i : sfimage.islands) {
        srclogger->info("");
        srclogger->info(" * Island {}: label={}, sign={}, extremum_val={}, extremum_x_idx={}, extremum_y_idy={}, num_samples={}",
            island_num, i.label_idx, i.sign, i.extremum_val, i.extremum_x_idx, i.extremum_y_idx, i.num_samples);
        srclogger->info("   Moments fit: amplitude={}, x_centre={}, y_centre={}, semimajor={}, semiminor={}, theta={}",
            i.moments_fit.amplitude, i.moments_fit.x_centre, i.moments_fit.y_centre, i.moments_fit.semimajor, i.moments_fit.semiminor, i.moments_fit.theta);
        if (print_gaussian_fit) {
            srclogger->info("   Bounding box: top={}, bottom={}, left={}, right={}, width={}, height={}",
                i.bounding_box.top, i.bounding_box.bottom, i.bounding_box.left, i.bounding_box.right, i.bounding_box.get_width(), i.bounding_box.get_height());
            srclogger->info("   Gaussian fit: amplitude={}, x_centre={}, y_centre={}, semimajor={}, semiminor={}, theta={}",
                i.leastsq_fit.amplitude, i.leastsq_fit.x_centre, i.leastsq_fit.y_centre, i.leastsq_fit.semimajor, i.leastsq_fit.semiminor, i.leastsq_fit.theta);
            srclogger->info("   {}", i.ceres_report);
        }
        island_num++;
    }
}

void log_function_timings()
{
    std::chrono::duration<double> time_span;
    benchlogger->info("");
    benchlogger->info("### FUNCTION TIMINGS [seconds] ###");

    if (!stp::times_iv.empty()) {
        // Imager
        std::vector<std::string> imager_steps = { "Grid init", "Grid loop", "FFT", "Normalise" };
        benchlogger->info(" Imager:");
        for (uint i = 1; i < stp::times_iv.size(); i++) {
            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_iv[i] - stp::times_iv[i - 1]);
            benchlogger->info(" - {:11s} = {:10.5f}", imager_steps[i - 1], time_span.count());
            if (i == 2 && !stp::times_gridder.empty()) {
                // Kernel generation
                benchlogger->info("   *Kernel Gen = {:10.5f}", stp::times_gridder[0].count());
            }
        }
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_iv.back() - stp::times_iv.front());
        benchlogger->info(" - Total       = {:10.5f}", time_span.count());
    }

    if (!stp::times_sf.empty()) {
        // Source find
        std::vector<std::string> sourcefind_steps = { "Bg level", "RMS est", "Labeling", "GaussianFit" };
        benchlogger->info(" Source find:");
        for (uint i = 1; i < stp::times_sf.size(); i++) {
            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_sf[i] - stp::times_sf[i - 1]);
            benchlogger->info(" - {:11s} = {:10.5f}", sourcefind_steps[i - 1], time_span.count());
        }
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_sf.back() - stp::times_sf.front());
        benchlogger->info(" - Total       = {:10.5f}", time_span.count());

        // Labeling
        std::vector<std::string> labeling_steps = { "Scanning", "Merging", "Analysis", "FinalLabels", "Measurement" };
        benchlogger->info(" Labeling:");
        for (uint i = 1; i < stp::times_ccl.size(); i++) {
            time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_ccl[i] - stp::times_ccl[i - 1]);
            benchlogger->info(" - {:11s} = {:10.5f}", labeling_steps[i - 1], time_span.count());
        }
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(stp::times_ccl.back() - stp::times_ccl.front());
        benchlogger->info(" - Total       = {:10.5f}", time_span.count());
    }

    // Global
    std::vector<std::string> reduce_steps;

    if (stp::times_iv.empty()) {
        reduce_steps = { "Read data", "Source find", "Write data" };
    } else {
        if (stp::times_sf.empty()) {
            reduce_steps = { "Read data", "Imager", "Write data" };
        } else {
            reduce_steps = { "Read data", "Imager", "Source find", "Write data" };
        }
    }

    benchlogger->info(" Global:");
    for (uint i = 1; i < times_main.size(); i++) {
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(times_main[i] - times_main[i - 1]);
        benchlogger->info(" - {:11s} = {:10.5f}", reduce_steps[i - 1], time_span.count());
    }
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(times_main.back() - times_main.front());
    benchlogger->info(" - Total       = {:10.5f}", time_span.count());
}
