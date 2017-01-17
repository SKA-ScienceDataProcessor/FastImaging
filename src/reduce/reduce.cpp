/**
* @file reduce.cpp
* Main file of reduce
* Contains the main function. Creates and configures the TCLAP interface.
* Call Convolutions/Kernel/Pipeline Funtions and prints the results.
*/

#include "reduce.h"

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
    // Adds the input filename flag to the parser list. This is flag is required
    _logger->debug("Adding the input filename flag to the command line parser");
    _cmd.add(_fileArg);

    // Adds the config json filename flag to the parse. This is flag is required
    _logger->debug("Adding the config json filename flag to the command line parser");
    _cmd.add(_confArg);
}

rapidjson::Document load_json_configuration(std::string& cfg)
{
    std::ifstream file(cfg);
    std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    rapidjson::Document aux;
    aux.Parse(str.c_str());

    return aux;
}

stp::source_find_image run_pipeline(arma::mat uvw_lambda, arma::cx_mat model_vis, arma::cx_mat data_vis, int image_size, double cell_size, bool kernel_exact, int oversampling, double detection_n_sigma, double analysis_n_sigma)
{
    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = data_vis - model_vis;

    // Kernel generation - might configure this via config-file in future.
    int kernel_support = 3;
    stp::GaussianSinc kernel_func(kernel_support);

    std::pair<arma::cx_mat, arma::cx_mat> result = stp::image_visibilities(kernel_func, residual_vis, uvw_lambda, image_size, cell_size, kernel_support, kernel_exact, oversampling);

    return stp::source_find_image(arma::real(result.first), detection_n_sigma, analysis_n_sigma, std::experimental::nullopt, true);
}

int main(int argc, char** argv)
{
    // Creates and initializes the logger
    initLogger();

    // Adds the flags to the parser
    createFlags();

    // Parses the arguments from the command console
    _cmd.parse(argc, argv);

    _logger->info("Loading data");
    // Load the input file argument to a npy array
    cnpy::NpyArray input_uvw1(cnpy::npz_load(_fileArg.getValue(), "uvw_lambda"));
    cnpy::NpyArray input_model1(cnpy::npz_load(_fileArg.getValue(), "model"));
    cnpy::NpyArray input_vis1(cnpy::npz_load(_fileArg.getValue(), "vis"));

    //Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(input_uvw1);
    arma::cx_mat input_model = load_npy_complex_array(input_model1);
    arma::cx_mat input_vis = load_npy_complex_array(input_vis1);

    // Load all configurations from json configuration file
    ConfigurationFile cfg(_confArg.getValue());

    _logger->info("Running pipeline");
    // Runs pipeline
    stp::source_find_image sfimage = run_pipeline(input_uvw, input_model, input_vis, cfg.image_size, cfg.cell_size, cfg.kernel_exact, cfg.oversampling, cfg.detection_n_sigma, cfg.analysis_n_sigma);

    _logger->info("Finished");

    int total_islands = sfimage.islands.size();
    _logger->info("Number of found islands: {} ", total_islands);

    for (int i = 0; i < total_islands; i++) {
        _logger->info(" * Island {}: label={}, sign={}, extremum_val={}, extremum_x_idx={}, extremum_y_idy={}, xbar={}, ybar={}",
            i, sfimage.islands[i].label_idx, sfimage.islands[i].sign, sfimage.islands[i].extremum_val, sfimage.islands[i].extremum_x_idx, sfimage.islands[i].extremum_y_idx,
            sfimage.islands[i].xbar, sfimage.islands[i].ybar);
    }

    return 0;
}
