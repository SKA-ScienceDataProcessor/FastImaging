/**
* @file stp_runner.cpp
* Main file of the stp runner
* Contains the main function. Creates and configures the TCLAP and Spdlog interface.
* Call Convolutions/Kernel/Pipeline Funtions and prints the results.
*/

#include "stp_runner.h"

// Mode chosen type map
std::map<std::string, func> mode_choosed = {
    { "convolution", convolution },
    { "kernel", kernel },
    { "pipeline", pipeline },
    { "fileconfiguration", fileconfiguration }
};

// Convolution chosen type map
std::map<std::string, convolution_types> conv_types = {
    { "tophat", tophat },
    { "triangle", triangle },
    { "sinc", sinc },
    { "gaussian", gaussian },
    { "gaussian-sinc", gaussianSinc }
};

void initLogger() throw(TCLAP::ArgException)
{
    // Creates two spdlog sinks
    // One sink for the stdout and another for a file
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_sink_st>());
    sinks.push_back(std::make_shared<spdlog::sinks::simple_file_sink_st>("logfile.txt"));
    _logger = std::make_shared<spdlog::logger>("logger", begin(sinks), end(sinks));
}

void createFlags() throw(TCLAP::ArgException)
{
    // Adds the action flag to the parse.
    _logger->debug("Adding the mode flag to the command line parser");
    _cmd.add(_modeFlag);

    // Adds the convolution flag to the parse.
    _logger->debug("Adding the mode flag to the command line parser");
    _cmd.add(_convFlag);

    // Adds the value flag to the parser list. This is flag is required
    _logger->debug("Adding the filepath flag to the command line parser");
    _cmd.add(_fileArg);
}

rapidjson::Document load_json_configuration()
{
    std::ifstream file("configuration.json");
    std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    rapidjson::Document aux;
    aux.Parse(str.c_str());

    return aux;
}

void run_all_configurations(ConfigurationFile& cfg, arma::cx_mat input, arma::cx_mat input_vis, arma::cx_mat input_uvw)
{
    _logger->info("\n 1D Convolution \n");
    _logger->info("Tophat");
    TopHat tophat(cfg.half_base_width_tophat);
    tophat(arma::real(input)).print();

    _logger->info("Triangle");
    Triangle triangle(cfg.half_base_width_triangle, cfg.triangle_value_triangle);
    triangle(arma::real(input)).print();

    _logger->info("Sinc");
    Sinc sinc(cfg.width_normalization_sinc, cfg.trunc_sinc);
    sinc(arma::real(input)).print();

    _logger->info("Gaussian");
    Gaussian gaussian(cfg.width_gaussian, cfg.trunc_gaussian);
    gaussian(arma::real(input)).print();

    _logger->info("GaussianSinc");
    GaussianSinc gaussiansinc(cfg.width_normalization_gaussian_gaussiansinc, cfg.width_normalization_sinc_gaussiansinc, cfg.trunc_gaussiansinc);
    gaussiansinc(arma::real(input)).print();

    _logger->info("\n 2D Convolution - Kernel \n");
    _logger->info("Tophat");
    make_kernel_array(cfg.support, arma::real(input), cfg.kernel_oversampling, cfg.pad, cfg.normalize, tophat).print();

    _logger->info("Triangle");
    make_kernel_array(cfg.support, arma::real(input), cfg.kernel_oversampling, cfg.pad, cfg.normalize, triangle).print();

    _logger->info("Sinc");
    make_kernel_array(cfg.support, arma::real(input), cfg.kernel_oversampling, cfg.pad, cfg.normalize, sinc).print();

    _logger->info("Gaussian");
    make_kernel_array(cfg.support, arma::real(input), cfg.kernel_oversampling, cfg.pad, cfg.normalize, gaussian).print();

    _logger->info("GaussianSinc");
    make_kernel_array(cfg.support, arma::real(input), cfg.kernel_oversampling, cfg.pad, cfg.normalize, gaussiansinc).print();

    _logger->info("\n Pipeline \n");
    _logger->info("Tophat");
    image_visibilities(input_vis, input_uvw, cfg.image_size, cfg.cell_size, cfg.support, cfg.pipeline_oversampling, cfg.pad, cfg.normalize, tophat).print();

    _logger->info("Triangle");
    image_visibilities(input_vis, input_uvw, cfg.image_size, cfg.cell_size, cfg.support, cfg.pipeline_oversampling, cfg.pad, cfg.normalize, triangle).print();

    _logger->info("Sinc");
    image_visibilities(input_vis, input_uvw, cfg.image_size, cfg.cell_size, cfg.support, cfg.pipeline_oversampling, cfg.pad, cfg.normalize, sinc).print();

    _logger->info("Gaussian");
    image_visibilities(input_vis, input_uvw, cfg.image_size, cfg.cell_size, cfg.support, cfg.pipeline_oversampling, cfg.pad, cfg.normalize, gaussian).print();

    _logger->info("GaussianSinc");
    image_visibilities(input_vis, input_uvw, cfg.image_size, cfg.cell_size, cfg.support, cfg.pipeline_oversampling, cfg.pad, cfg.normalize, gaussiansinc).print();
}

void run_convolution(ConfigurationFile& cfg, arma::cx_mat input)
{
    switch (conv_types.at(_convFlag.getValue())) {
    case tophat: {
        _logger->info("Starting Tophat");
        TopHat(cfg.half_base_width_tophat)(arma::real(input)).print();
    } break;
    case triangle: {
        _logger->info("Starting Triangle");
        Triangle(cfg.half_base_width_triangle, cfg.triangle_value_triangle)(arma::real(input)).print();
    } break;
    case sinc: {
        _logger->info("Starting Sinc");
        Sinc(cfg.width_normalization_sinc, cfg.trunc_sinc)(arma::real(input)).print();
    } break;
    case gaussian: {
        _logger->info("Starting Gaussian");
        Gaussian(cfg.width_gaussian, cfg.trunc_gaussian)(arma::real(input)).print();
    } break;
    case gaussianSinc: {
        _logger->info("Starting GaussianSinc");
        GaussianSinc(cfg.width_normalization_gaussian_gaussiansinc, cfg.width_normalization_sinc_gaussiansinc, cfg.trunc_gaussiansinc)(arma::real(input)).print();
    } break;
    }
}

void run_kernel(ConfigurationFile& cfg, arma::cx_mat input)
{
    switch (conv_types.at(_convFlag.getValue())) {
    case tophat: {
        _logger->info("Starting Tophat");
        make_kernel_array(cfg.support, arma::real(input), cfg.kernel_oversampling, cfg.pad, cfg.normalize, TopHat(cfg.half_base_width_tophat)).print();
    } break;
    case triangle: {
        _logger->info("Starting Triangle");
        make_kernel_array(cfg.support, arma::real(input), cfg.kernel_oversampling, cfg.pad, cfg.normalize, Triangle(cfg.half_base_width_triangle, cfg.triangle_value_triangle)).print();
    } break;
    case sinc: {
        _logger->info("Starting Sinc");
        make_kernel_array(cfg.support, arma::real(input), cfg.kernel_oversampling, cfg.pad, cfg.normalize, Sinc(cfg.width_normalization_sinc, cfg.trunc_sinc)).print();
    } break;
    case gaussian: {
        _logger->info("Starting Gaussian");
        make_kernel_array(cfg.support, arma::real(input), cfg.kernel_oversampling, cfg.pad, cfg.normalize, Gaussian(cfg.width_gaussian, cfg.trunc_gaussian)).print();
    } break;
    case gaussianSinc: {
        _logger->info("Starting GaussianSinc");
        make_kernel_array(cfg.support, arma::real(input), cfg.kernel_oversampling, cfg.pad, cfg.normalize, GaussianSinc(cfg.width_normalization_gaussian_gaussiansinc, cfg.width_normalization_sinc_gaussiansinc, cfg.trunc_gaussiansinc)).print();
    } break;
    }
}

void run_pipeline(ConfigurationFile& cfg, arma::cx_mat input_vis, arma::cx_mat input_uvw)
{
    switch (conv_types.at(_convFlag.getValue())) {
    case tophat: {
        _logger->info("Starting Tophat");
        image_visibilities(input_vis, input_uvw, cfg.image_size, cfg.cell_size, cfg.support, cfg.pipeline_oversampling, cfg.pad, cfg.normalize, TopHat(cfg.half_base_width_tophat)).print();
    } break;
    case triangle: {
        _logger->info("Starting Triangle");
        image_visibilities(input_vis, input_uvw, cfg.image_size, cfg.cell_size, cfg.support, cfg.pipeline_oversampling, cfg.pad, cfg.normalize, Triangle(cfg.half_base_width_triangle, cfg.triangle_value_triangle)).print();
    } break;
    case sinc: {
        _logger->info("Starting Sinc");
        image_visibilities(input_vis, input_uvw, cfg.image_size, cfg.cell_size, cfg.support, cfg.pipeline_oversampling, cfg.pad, cfg.normalize, Sinc(cfg.width_normalization_sinc, cfg.trunc_sinc)).print();
    } break;
    case gaussian: {
        _logger->info("Starting Gaussian");
        image_visibilities(input_vis, input_uvw, cfg.image_size, cfg.cell_size, cfg.support, cfg.pipeline_oversampling, cfg.pad, cfg.normalize, Gaussian(cfg.width_gaussian, cfg.trunc_gaussian)).print();
    } break;
    case gaussianSinc: {
        _logger->info("Starting GaussianSinc");
        image_visibilities(input_vis, input_uvw, cfg.image_size, cfg.cell_size, cfg.support, cfg.pipeline_oversampling, cfg.pad, cfg.normalize, GaussianSinc(cfg.width_normalization_gaussian_gaussiansinc, cfg.width_normalization_sinc_gaussiansinc, cfg.trunc_gaussiansinc)).print();
    } break;
    }
}

int main(int argc, char** argv)
{
    // Creates and initializes the logger
    initLogger();
    _logger->info("Program start");

    // Adds the flags to the parser
    createFlags();

    // Parses the arguments from the command console
    _logger->info("Parsing arguments from console");
    _cmd.parse(argc, argv);

    // Load the file argument to a npy array
    cnpy::NpyArray input_vis1(cnpy::npz_load(_fileArg.getValue(), "vis"));
    cnpy::NpyArray input_uvw1(cnpy::npz_load(_fileArg.getValue(), "uvw"));

    arma::cx_mat input_vis = load_npy_array(&input_vis1);
    arma::cx_mat input_uvw = load_npy_array(&input_uvw1);

    arma::cx_mat input;
    // Load all configurations from json configuration file
    ConfigurationFile cfg;

    if (cfg.type_input == "vis") {
        input = input_vis;
    } else if (cfg.type_input == "uvw") {
        input = input_uvw;
    }

    // Chooses the mode to run
    try {
        switch (mode_choosed.at(_modeFlag.getValue())) {
        case convolution: {
            run_convolution(cfg, input);
        } break;
        case kernel: {
            run_kernel(cfg, input);
        } break;
        case pipeline: {
            run_pipeline(cfg, input_vis, input_uvw);
        } break;
        case fileconfiguration: {
            run_all_configurations(cfg, input, input_vis, input_uvw);
        } break;
        }
    } catch (std::out_of_range& e) {
        _logger->error("Incorrect mode choosen");
        return program_finish;
    }
    _logger->info("Program ends");
    return program_finish;
}
