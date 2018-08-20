/** @file pipeline_handler.h
 *  @brief Header file for the pipeline tests
 *
 *  Reads the configuration json, loads test data onto variables, and executes pipeline
 */

#include <load_data.h>
#include <load_json_config.h>
#include <stp.h>

class SlowTransientPipeline {

public:
    //Constructors
    SlowTransientPipeline();
    SlowTransientPipeline(std::string& datafile, std::string& configfile);

    // Functions for pipeline execution
    void load_testdata(std::string& datafile);
    void load_configdata(std::string& configfile);
    stp::SourceFindImage execute_pipeline();

    // Simulated data
    arma::mat input_uvw;
    arma::cx_mat input_vis;
    arma::mat input_snr_weights;
    arma::mat skymodel;
    arma::cx_mat residual_vis;

    // Configuration parameters
    stp::ImagerPars img_pars;
    stp::W_ProjectionPars w_proj;
    stp::A_ProjectionPars a_proj;
    double detection_n_sigma;
    double analysis_n_sigma;
    double rms_estimation;
    int sigma_clip_iters;
    stp::MedianMethod median_method;
    bool gaussian_fitting;
    bool ccl_4connectivity;
    int source_min_area;
    bool generate_labelmap;
    stp::CeresDiffMethod ceres_diffmethod;
    stp::CeresSolverType ceres_solvertype;
    bool find_negative_sources;
};

SlowTransientPipeline::SlowTransientPipeline()
{
    ConfigurationFile cfg;
    img_pars = cfg.img_pars;
    w_proj = cfg.w_proj;
    a_proj = cfg.a_proj;
    detection_n_sigma = cfg.detection_n_sigma;
    analysis_n_sigma = cfg.analysis_n_sigma;
    rms_estimation = cfg.estimate_rms;
    sigma_clip_iters = cfg.sigma_clip_iters;
    median_method = cfg.median_method;
    gaussian_fitting = cfg.gaussian_fitting;
    ccl_4connectivity = cfg.ccl_4connectivity;
    source_min_area = cfg.source_min_area;
    generate_labelmap = cfg.generate_labelmap;
    ceres_diffmethod = cfg.ceres_diffmethod;
    ceres_solvertype = cfg.ceres_solvertype;
    find_negative_sources = cfg.find_negative_sources;
}

SlowTransientPipeline::SlowTransientPipeline(std::string& datafile, std::string& configfile)
{
    load_testdata(datafile);
    load_configdata(configfile);
}

void SlowTransientPipeline::load_testdata(std::string& datafile)
{
    // Load simulated data from input_npz
    input_uvw = load_npy_double_array<double>(datafile, "uvw_lambda");
    input_vis = load_npy_complex_array<double>(datafile, "vis");
    input_snr_weights = load_npy_double_array<double>(datafile, "snr_weights");
    skymodel = load_npy_double_array<double>(datafile, "skymodel");

    // Generate model visibilities from the skymodel and UVW-baselines
    arma::cx_mat model_vis = stp::generate_visibilities_from_local_skymodel(skymodel, input_uvw);

    // Subtract model-generated visibilities from incoming data
    residual_vis = input_vis - model_vis;
}

void SlowTransientPipeline::load_configdata(std::string& configfile)
{
    ConfigurationFile cfg(configfile);
    img_pars = cfg.img_pars;
    w_proj = cfg.w_proj;
    a_proj = cfg.a_proj;
    detection_n_sigma = cfg.detection_n_sigma;
    analysis_n_sigma = cfg.analysis_n_sigma;
    rms_estimation = cfg.estimate_rms;
    sigma_clip_iters = cfg.sigma_clip_iters;
    median_method = cfg.median_method;
    gaussian_fitting = cfg.gaussian_fitting;
    ccl_4connectivity = cfg.ccl_4connectivity;
    source_min_area = cfg.source_min_area;
    generate_labelmap = cfg.generate_labelmap;
    ceres_diffmethod = cfg.ceres_diffmethod;
    ceres_solvertype = cfg.ceres_solvertype;
    find_negative_sources = cfg.find_negative_sources;
}

stp::SourceFindImage SlowTransientPipeline::execute_pipeline()
{
    //Imager
    stp::ImageVisibilities imager(std::move(residual_vis), std::move(input_snr_weights),
        std::move(input_uvw), img_pars, w_proj, a_proj);

    // Sourcefind
    return stp::SourceFindImage(imager.vis_grid, detection_n_sigma, analysis_n_sigma, rms_estimation, find_negative_sources,
        sigma_clip_iters, median_method, gaussian_fitting, ccl_4connectivity, generate_labelmap, source_min_area,
        ceres_diffmethod, ceres_solvertype);
}
