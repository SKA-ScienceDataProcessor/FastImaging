#ifndef LOAD_JSON_CONFIG_H
#define LOAD_JSON_CONFIG_H

#include "../stp/types.h"
#include <fstream>

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
    // ConfigurationFile default constuctor
    ConfigurationFile() = default;

    /**
     * @brief ConfigurationFile constructor
     *
     * Load all parameters in json configuration file to an object
     *
     * @param[in] cfg (string) : Input config json filename
     */
    ConfigurationFile(const std::string& cfg)
    {
        document = load_json_configuration(cfg);

        if (document.IsObject()) {

            // Imager settings
            rapidjson::Value::ConstMemberIterator secitr = document.FindMember("imager_settings");
            if (secitr != document.MemberEnd()) {
                rapidjson::Value::ConstMemberIterator itr = secitr->value.FindMember("image_size_pix");
                if (itr != secitr->value.MemberEnd())
                    img_pars.image_size = itr->value.GetInt();
                itr = secitr->value.FindMember("cell_size_arcsec");
                if (itr != secitr->value.MemberEnd())
                    img_pars.cell_size = itr->value.GetDouble();
                itr = secitr->value.FindMember("kernel_function");
                if (itr != secitr->value.MemberEnd()) {
                    s_kernel_function = itr->value.GetString();
                    img_pars.kernel_function = parse_kernel_function(s_kernel_function);
                }
                itr = secitr->value.FindMember("kernel_support");
                if (itr != secitr->value.MemberEnd())
                    img_pars.kernel_support = itr->value.GetInt();
                itr = secitr->value.FindMember("kernel_exact");
                if (itr != secitr->value.MemberEnd())
                    img_pars.kernel_exact = itr->value.GetBool();
                itr = secitr->value.FindMember("oversampling");
                if (itr != secitr->value.MemberEnd())
                    img_pars.oversampling = itr->value.GetInt();
                itr = secitr->value.FindMember("generate_beam");
                if (itr != secitr->value.MemberEnd())
                    img_pars.generate_beam = itr->value.GetBool();
                itr = secitr->value.FindMember("gridding_correction");
                if (itr != secitr->value.MemberEnd())
                    img_pars.gridding_correction = itr->value.GetBool();
                itr = secitr->value.FindMember("analytic_gcf");
                if (itr != secitr->value.MemberEnd())
                    img_pars.analytic_gcf = itr->value.GetBool();
                itr = secitr->value.FindMember("fft_routine");
                if (itr != secitr->value.MemberEnd()) {
                    s_fft_routine = itr->value.GetString();
                    img_pars.r_fft = parse_fft_routine(s_fft_routine);
                }
                itr = secitr->value.FindMember("fft_wisdom_filename");
                if (itr != secitr->value.MemberEnd())
                    img_pars.fft_wisdom_filename = itr->value.GetString();
            }
#ifdef WPROJECTION
            // W-Projection settings
            secitr = document.FindMember("wprojection_settings");
            if (secitr != document.MemberEnd()) {
                rapidjson::Value::ConstMemberIterator itr = secitr->value.FindMember("num_wplanes");
                if (itr != secitr->value.MemberEnd())
                    w_proj.num_wplanes = itr->value.GetInt();
                itr = secitr->value.FindMember("max_wpconv_support");
                if (itr != secitr->value.MemberEnd())
                    w_proj.max_wpconv_support = itr->value.GetInt();
                itr = secitr->value.FindMember("undersampling_opt");
                if (itr != secitr->value.MemberEnd())
                    w_proj.undersampling_opt = itr->value.GetInt();
                itr = secitr->value.FindMember("kernel_trunc_perc");
                if (itr != secitr->value.MemberEnd())
                    w_proj.kernel_trunc_perc = itr->value.GetDouble();
                itr = secitr->value.FindMember("hankel_opt");
                if (itr != secitr->value.MemberEnd())
                    w_proj.hankel_opt = itr->value.GetBool();
                itr = secitr->value.FindMember("interp_type");
                if (itr != secitr->value.MemberEnd()) {
                    s_interp_type = itr->value.GetString();
                    w_proj.interp_type = parse_interp_type(s_interp_type);
                }
                itr = secitr->value.FindMember("wplanes_median");
                if (itr != secitr->value.MemberEnd())
                    w_proj.wplanes_median = itr->value.GetBool();
            }
#endif
#ifdef APROJECTION
            // A-Projection settings
            secitr = document.FindMember("aprojection_settings");
            if (secitr != document.MemberEnd()) {
                rapidjson::Value::ConstMemberIterator itr = secitr->value.FindMember("aproj_numtimesteps");
                if (itr != secitr->value.MemberEnd())
                    a_proj.num_timesteps = itr->value.GetDouble();
                itr = secitr->value.FindMember("obs_dec");
                if (itr != secitr->value.MemberEnd())
                    a_proj.obs_dec = itr->value.GetDouble();
                itr = secitr->value.FindMember("obs_lat");
                if (itr != secitr->value.MemberEnd())
                    a_proj.obs_lat = itr->value.GetDouble();
            }
#endif
            // Source Find settings
            secitr = document.FindMember("sourcefind_settings");
            if (secitr != document.MemberEnd()) {
                rapidjson::Value::ConstMemberIterator itr = secitr->value.FindMember("sourcefind_detection");
                if (itr != secitr->value.MemberEnd())
                    detection_n_sigma = itr->value.GetDouble();
                itr = secitr->value.FindMember("sourcefind_analysis");
                if (itr != secitr->value.MemberEnd())
                    analysis_n_sigma = itr->value.GetDouble();
                itr = secitr->value.FindMember("find_negative_sources");
                if (itr != secitr->value.MemberEnd())
                    find_negative_sources = itr->value.GetBool();
                itr = secitr->value.FindMember("rms_estimation");
                if (itr != secitr->value.MemberEnd())
                    estimate_rms = itr->value.GetDouble();
                itr = secitr->value.FindMember("sigma_clip_iters");
                if (itr != secitr->value.MemberEnd())
                    sigma_clip_iters = itr->value.GetInt();
                itr = secitr->value.FindMember("median_method");
                if (itr != secitr->value.MemberEnd()) {
                    s_median_method = itr->value.GetString();
                    median_method = parse_median_method(s_median_method);
                }
                itr = secitr->value.FindMember("gaussian_fitting");
                if (itr != secitr->value.MemberEnd())
                    gaussian_fitting = itr->value.GetBool();
                itr = secitr->value.FindMember("ccl_4connectivity");
                if (itr != secitr->value.MemberEnd())
                    ccl_4connectivity = itr->value.GetBool();
                itr = secitr->value.FindMember("generate_labelmap");
                if (itr != secitr->value.MemberEnd())
                    generate_labelmap = itr->value.GetBool();
                itr = secitr->value.FindMember("source_min_area");
                if (itr != secitr->value.MemberEnd())
                    source_min_area = itr->value.GetInt();
                itr = secitr->value.FindMember("ceres_diffmethod");
                if (itr != secitr->value.MemberEnd()) {
                    s_ceres_diffmethod = itr->value.GetString();
                    ceres_diffmethod = parse_ceres_diffmethod(s_ceres_diffmethod);
                }
                itr = secitr->value.FindMember("ceres_solvertype");
                if (itr != secitr->value.MemberEnd()) {
                    s_ceres_solvertype = itr->value.GetString();
                    ceres_solvertype = parse_ceres_solvertype(s_ceres_solvertype);
                }
            }
        } else {
            assert(0);
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

    // Imager settings
    stp::ImagerPars img_pars;
    stp::W_ProjectionPars w_proj;
    stp::A_ProjectionPars a_proj;
    std::string s_kernel_function = "PSWF";
    std::string s_fft_routine = "FFTW_ESTIMATE_FFT";
    std::string s_interp_type = "linear";

    // Source find settings
    double detection_n_sigma = 0.0;
    double analysis_n_sigma = 0.0;
    bool find_negative_sources = true;
    double estimate_rms = 0.0;
    int sigma_clip_iters = 5;
    std::string s_median_method = "BINMEDIAN";
    stp::MedianMethod median_method = stp::MedianMethod::BINMEDIAN;
    bool gaussian_fitting = true;
    bool ccl_4connectivity = false;
    bool generate_labelmap = false;
    std::string s_ceres_diffmethod = "AutoDiff_SingleResBlk";
    std::string s_ceres_solvertype = "LinearSearch_BFGS";
    stp::CeresDiffMethod ceres_diffmethod = stp::CeresDiffMethod::AutoDiff_SingleResBlk;
    stp::CeresSolverType ceres_solvertype = stp::CeresSolverType::LinearSearch_BFGS;
    int source_min_area = 5;

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
     * @brief Parse string of the interpolation type
     *
     * @param[in] it (string): Input interpolation type string
     *
     * @return (InterpType) Enumeration value for the input interpolation type
     */
    stp::InterpType parse_interp_type(const std::string& it);

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
    rapidjson::Document document;
};

#endif /* LOAD_JSON_CONFIG_H */
