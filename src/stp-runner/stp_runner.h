/**
* @file stp_runner.h
* Header file for the stp runner
* Containts the includes, function declarations, global shared variables to be
* used in the stp runner and the struct to load the configurations.
*/

#ifndef STP_RUNNER_H
#define STP_RUNNER_H

#include <load_data.h>

#include <libstp.h>

// Spdlog
#include <spdlog/spdlog.h>

// TCLAP
#include <tclap/CmdLine.h>

// RapidJson
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

// Constants
const char program_finish(1);

// Function types
enum func {
    convolution,
    kernel,
    pipeline,
    fileconfiguration
};
enum convolution_types {
    tophat,
    triangle,
    sinc,
    gaussian,
    gaussianSinc
};

// Logger
std::shared_ptr<spdlog::logger> _logger;
// Mode flag
TCLAP::ValueArg<std::string> _modeFlag("m", "mode", "Chooses which mode to use\n convolution for 1D convolution\nkernel for 2D convolution\npipeline for pipeline convolution\nfileconfiguration to load all configurations/calculate from configuration file", true, "fileconfiguration", "string");
// Convolution flag
TCLAP::ValueArg<std::string> _convFlag("c", "convolution-types", "Chooses which function to use\ntophat for tophat\nsinc for sinc\ngaussian for gaussian\ntriangle for triangle\ngaussian-sinc for gaussian-sinc\n", false, "tophat", "string");
// Filepath flag
TCLAP::ValueArg<std::string> _fileArg("f", "filepath", "Input filepath", true, "../mock_uvw_vis.npz", "string");
// Command line parser
TCLAP::CmdLine _cmd("Parser for the stp runner", ' ', "0.3");

/**
 * @brief Loads a json configuration file into a rapidjson document
 *
 * Uses ifstream to load the json file, and after this parses the result into a document.
 *
 * @return A rapidjson::Document with the content of a json file.
 */
rapidjson::Document load_json_configuration();

/**
 * @brief The Configuration file struct
 *
 * Load all parameters in json configuration file to an object
 */
struct ConfigurationFile {
    rapidjson::Document config_document;
    rapidjson::Value tophat;
    rapidjson::Value triangle;
    rapidjson::Value sinc;
    rapidjson::Value gaussian;
    rapidjson::Value gaussiansinc;
    rapidjson::Value kernel;
    rapidjson::Value pipeline;

    bool pad;
    bool normalize;
    int support;
    double half_base_width_tophat;
    double half_base_width_triangle;
    double triangle_value_triangle;
    double kernel_oversampling;
    double pipeline_oversampling;
    double cell_size;
    double image_size;
    double width_normalization_sinc;
    double trunc_sinc;
    double width_gaussian;
    double trunc_gaussian;
    double width_normalization_gaussian_gaussiansinc;
    double width_normalization_sinc_gaussiansinc;
    double trunc_gaussiansinc;
    std::string type_input;

public:
    ConfigurationFile()
    {
        config_document = load_json_configuration();

        tophat = config_document["convolution"]["tophat"];
        triangle = config_document["convolution"]["triangle"];
        sinc = config_document["convolution"]["sinc"];
        gaussian = config_document["convolution"]["gaussian"];
        gaussiansinc = config_document["convolution"]["gaussiansinc"];
        kernel = config_document["kernel"];
        pipeline = config_document["pipeline"];

        half_base_width_tophat = tophat["half_base_width"].GetDouble();
        half_base_width_triangle = triangle["half_base_width"].GetDouble();
        triangle_value_triangle = triangle["triangle_value"].GetDouble();

        pad = kernel["pad"].GetBool();
        normalize = kernel["normalize"].GetBool();
        kernel_oversampling = kernel["oversampling"].GetDouble();
        support = kernel["oversampling"].GetInt();

        pipeline_oversampling = pipeline["oversampling"].GetDouble();
        cell_size = pipeline["cell_size"].GetDouble();
        image_size = pipeline["image_size"].GetDouble();

        width_normalization_sinc = sinc["width_normalization"].GetDouble();
        trunc_sinc = sinc["trunc"].GetDouble();

        width_gaussian = gaussian["gaussian_width"].GetDouble();
        trunc_gaussian = gaussian["trunc"].GetDouble();

        width_normalization_gaussian_gaussiansinc = gaussiansinc["width_normalization_gaussian"].GetDouble();
        width_normalization_sinc_gaussiansinc = gaussiansinc["width_normalization_sinc"].GetDouble();
        trunc_gaussiansinc = gaussiansinc["trunc"].GetDouble();

        type_input = config_document["type_input"].GetString();
    }
};

/**
* @brief Logger initialization function
*
* Creates and initializes the logger to be used throughout the program
*/
void initLogger() throw(TCLAP::ArgException);

/**
* @brief Creates the switch flags to be used by the parser
*
* Creates a list of switch arguments and then xor adds to the command line parser, meaning that only
* one flag and one only is required to run the program_invocation_name
*
*/
void createFlags() throw(TCLAP::ArgException);

/**
 * @brief Run all convolutions classes
 *
 * Uses the configuration parameters and inputed matrices to call all classes.
 */
void run_all_configurations(ConfigurationFile& cfg, arma::cx_mat input, arma::cx_mat input_vis, arma::cx_mat input_uvw);

/**
 * @brief Run convolution
 *
 * Uses the configuration parameters/inputed matrix and command
 * line parameter to call the respective class and prints the result.
 */
void run_convolution(ConfigurationFile& cfg, arma::cx_mat input);

/**
 * @brief Run kernel
 *
 * Uses the configuration parameters/inputed matrix and command line parameter
 * to call the respective class and prints the kernel result.
 */
void run_kernel(ConfigurationFile& cfg, arma::cx_mat input);

/**
 * @brief Run pipeline
 *
 * Uses the configuration parameters/inputed matrices and command line parameter
 * to call the respective class and prints the pipeline result.
 */
void run_pipeline(ConfigurationFile& cfg, arma::cx_mat input_vis, arma::cx_mat input_uv);

#endif /* STP_RUNNER_H */
