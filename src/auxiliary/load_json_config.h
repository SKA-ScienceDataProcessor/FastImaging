#ifndef LOAD_JSON_CONFIG_H
#define LOAD_JSON_CONFIG_H

#include <fstream>
#include <stp.h>

// RapidJson
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

/**
 * @brief The ConfigurationFile struct
 *
 * Load all parameters in json configuration file to an object
 */
class ConfigurationFile {

public:
    // Delete ConfigurationFile default constuctor
    ConfigurationFile() = delete;

    /**
     * @brief ConfigurationFile constructor
     *
     * Load all parameters in json configuration file to an object
     *
     * @param[in] cfg (string) : Input config json filename
     */
    ConfigurationFile(const std::string& cfg)
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
            if (config_document.HasMember("find_negative_sources"))
                find_negative_sources = config_document["find_negative_sources"].GetBool();

            if (config_document.HasMember("kernel_func")) {
                s_kernel_function = config_document["kernel_func"].GetString();
                kernel_function = parse_kernel_function(s_kernel_function);
            }
            if (config_document.HasMember("fft_routine")) {
                s_fft_routine = config_document["fft_routine"].GetString();
                fft_routine = parse_fft_routine(s_fft_routine);
            }
            if (config_document.HasMember("fft_wisdom_filename"))
                fft_wisdom_filename = config_document["fft_wisdom_filename"].GetString();

            if (config_document.HasMember("rms_estimation"))
                estimate_rms = config_document["rms_estimation"].GetDouble();
            if (config_document.HasMember("sigma_clip_iters"))
                sigma_clip_iters = config_document["sigma_clip_iters"].GetInt();
            if (config_document.HasMember("median_method")) {
                s_median_method = config_document["median_method"].GetString();
                median_method = parse_median_method(s_median_method);
            }

            if (config_document.HasMember("gaussian_fitting"))
                gaussian_fitting = config_document["gaussian_fitting"].GetBool();
            if (config_document.HasMember("generate_labelmap"))
                generate_labelmap = config_document["generate_labelmap"].GetBool();
            if (config_document.HasMember("generate_beam"))
                generate_beam = config_document["generate_beam"].GetBool();

            if (config_document.HasMember("ceres_diffmethod")) {
                s_ceres_diffmethod = config_document["ceres_diffmethod"].GetString();
                ceres_diffmethod = parse_ceres_diffmethod(s_ceres_diffmethod);
            }
            if (config_document.HasMember("ceres_solvertype")) {
                s_ceres_solvertype = config_document["ceres_solvertype"].GetString();
                ceres_solvertype = parse_ceres_solvertype(s_ceres_solvertype);
            }
        }
    }

    /**
     * @brief Loads a json configuration file into a rapidjson document
     *
     * Uses ifstream to load the json file, and after this parses the result into a document.
     *
     * @param[in] cfg (string): Input config json filename.
     *
     * @return A rapidjson::Document with the content of a json file.
     */
    rapidjson::Document load_json_configuration(const std::string& cfg);

    int image_size = 0;
    double cell_size = 0.0;
    int kernel_support = 3;
    bool kernel_exact = false;
    int oversampling = 9;
    double detection_n_sigma = 0.0;
    double analysis_n_sigma = 0.0;
    bool find_negative_sources = true;
    std::string s_kernel_function = "GaussianSinc";
    std::string s_fft_routine = "FFTW_ESTIMATE_FFT";
    stp::KernelFunction kernel_function = stp::KernelFunction::GaussianSinc;
    stp::FFTRoutine fft_routine = stp::FFTRoutine::FFTW_ESTIMATE_FFT;
    std::string fft_wisdom_filename;
    double estimate_rms = 0.0;
    int sigma_clip_iters = 5;
    std::string s_median_method = "BINAPPROX";
    stp::MedianMethod median_method = stp::MedianMethod::BINAPPROX;
    bool gaussian_fitting = true;
    bool generate_labelmap = false;
    bool generate_beam = false;
    std::string s_ceres_diffmethod = "AutoDiff_SingleResBlk";
    std::string s_ceres_solvertype = "LinearSearch_BFGS";
    stp::CeresDiffMethod ceres_diffmethod = stp::CeresDiffMethod::AutoDiff_SingleResBlk;
    stp::CeresSolverType ceres_solvertype = stp::CeresSolverType::LinearSearch_BFGS;

private:
    /**
     * @brief Parse string of kernel function
     *
     * @param[in] kernel (string): Input kernel function string
     *
     * @return (KernelFunction) Enumeration value for the input kernel function
     */
    stp::KernelFunction parse_kernel_function(const std::string& kernel);

    /**
     * @brief Parse string of fft routine
     *
     * @param[in] fft (string): Input fft routine string
     *
     * @return (FFTRoutine) Enumeration value for the input fft routine
     */
    stp::FFTRoutine parse_fft_routine(const std::string& fft);

    /**
     * @brief Parse string of median method
     *
     * @param[in] medianmethod (string): Input median method string
     *
     * @return (MedianMethod) Enumeration value for the input median method
     */
    stp::MedianMethod parse_median_method(const std::string& medianmethod);

    /**
     * @brief Parse differentiation method used by ceres
     *
     * @param[in] diffmet (string): Input ceres differentiation method
     *
     * @return (CeresDiffMethod) Enumeration value for the ceres differentiation method
     */
    stp::CeresDiffMethod parse_ceres_diffmethod(const std::string& diffmet);

    /**
     * @brief Parse solver type used by ceres
     *
     * @param[in] solvertype (string): Input ceres solver type
     *
     * @return (CeresSolverType) Enumeration value for the ceres solver type
     */
    stp::CeresSolverType parse_ceres_solvertype(const std::string& solvertype);

    /**
     * Stores content from the input json file
     */
    rapidjson::Document config_document;
};

#endif /* LOAD_JSON_CONFIG_H */
