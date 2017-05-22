#ifndef LOAD_JSON_CONFIG_H
#define LOAD_JSON_CONFIG_H

#include <fstream>
#include <stp.h>

// RapidJson
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

/**
 * @brief The KernelFunction enum
 */
enum struct KernelFunction {
    TopHat = 0,
    Triangle,
    Sinc,
    Gaussian,
    GaussianSinc
};

/**
 * @brief Loads a json configuration file into a rapidjson document
 *
 * Uses ifstream to load the json file, and after this parses the result into a document.
 *
 * @param[in] cfg (string): Input config json filename.
 *
 * @return A rapidjson::Document with the content of a json file.
 */
rapidjson::Document
load_json_configuration(std::string& cfg);

/**
 * @brief The Configuration file struct
 *
 * Load all parameters in json configuration file to an object
 */
struct ConfigurationFile {
    rapidjson::Document config_document;

    int image_size = 0;
    double cell_size = 0.0;
    int kernel_support = 3;
    bool kernel_exact = false;
    int oversampling = 9;
    double detection_n_sigma = 0.0;
    double analysis_n_sigma = 0.0;
    std::string kernel_func = "GaussianSinc";
    std::string fft_routine = "FFTW_ESTIMATE_FFT";
    std::string image_wisdom_filename;
    std::string beam_wisdom_filename;
    bool gen_fullbeam = true;
    double estimate_rms = 0.0;
    bool compute_bg_level = true;
    bool compute_barycentre = true;
    int sigma_clip_iters = 5;

public:
    ConfigurationFile() = delete;

    ConfigurationFile(std::string cfg)
    {
        config_document = load_json_configuration(cfg);

        if (config_document.IsObject()) {
            if (config_document.HasMember("image_size_pix"))
                image_size = config_document["image_size_pix"].GetInt();
            if (config_document.HasMember("cell_size_arcsec"))
                cell_size = config_document["cell_size_arcsec"].GetDouble();
            if (config_document.HasMember("kernel_support"))
                kernel_support = config_document["kernel_support"].GetInt();
            if (config_document.HasMember("kernel_exact"))
                kernel_exact = config_document["kernel_exact"].GetBool();
            if (config_document.HasMember("oversampling"))
                oversampling = config_document["oversampling"].GetInt();
            if (config_document.HasMember("sourcefind_detection"))
                detection_n_sigma = config_document["sourcefind_detection"].GetDouble();
            if (config_document.HasMember("sourcefind_analysis"))
                analysis_n_sigma = config_document["sourcefind_analysis"].GetDouble();

            if (config_document.HasMember("kernel_func"))
                kernel_func = config_document["kernel_func"].GetString();
            if (config_document.HasMember("fft_routine"))
                fft_routine = config_document["fft_routine"].GetString();
            if (config_document.HasMember("image_fft_wisdom"))
                image_wisdom_filename = config_document["image_fft_wisdom"].GetString();
            if (config_document.HasMember("beam_fft_wisdom"))
                beam_wisdom_filename = config_document["beam_fft_wisdom"].GetString();
            if (config_document.HasMember("generate_full_beam"))
                gen_fullbeam = config_document["generate_full_beam"].GetBool();
            if (config_document.HasMember("estimate_rms"))
                estimate_rms = config_document["estimate_rms"].GetDouble();
            if (config_document.HasMember("compute_bg_level"))
                compute_bg_level = config_document["compute_bg_level"].GetBool();

            if (config_document.HasMember("compute_barycentre"))
                compute_barycentre = config_document["compute_barycentre"].GetBool();
            if (config_document.HasMember("sigma_clip_iterations"))
                sigma_clip_iters = config_document["sigma_clip_iterations"].GetInt();
        }
    }
};

/**
 * @brief Parse string of kernel function
 *
 * @param[in] kernel (string): Input kernel function string
 *
 * @return (KernelFunction) Enumeration value for the input kernel function
 */
KernelFunction parse_kernel_function(const std::string& kernel);

/**
 * @brief Parse string of fft routine
 *
 * @param[in] fft (string): Input fft routine string
 *
 * @return (fft_routine) Enumeration value for the input fft routine
 */
stp::fft_routine parse_fft_routine(const std::string& fft);

#endif /* LOAD_JSON_CONFIG_H */
