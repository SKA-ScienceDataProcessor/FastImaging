#include <gtest/gtest.h>
#include <load_data.h>
#include <load_json_config.h>
#include <stp.h>

std::string config_path(_PIPELINE_CONFIGPATH);
std::string data_path(_PIPELINE_DATAPATH);
std::string input_npz("simdata_nstep10.npz");

// As the results are compared with python output (which uses double type) the float case presents larger error
#ifdef USE_FLOAT
const double pipeline_tolerance(1.0e-4);
#else
const double pipeline_tolerance(1.0e-12);
#endif

class PipelineGaussianSincTest : public ::testing::Test {
public:
    // Overload SetUp function which receives configuration filename.
    // Must be explicitly called in the tests.
    void SetUp(const std::string& file)
    {
        ConfigurationFile cfg(file);
        image_size = cfg.image_size;
        cell_size = cfg.cell_size;
        kernel_support = cfg.kernel_support;
        kernel_exact = cfg.kernel_exact;
        oversampling = cfg.oversampling;
        detection_n_sigma = cfg.detection_n_sigma;
        analysis_n_sigma = cfg.analysis_n_sigma;
        kernel_function = cfg.kernel_function;
        fft_routine = cfg.fft_routine;
        rms_estimation = cfg.estimate_rms;
        sigma_clip_iters = cfg.sigma_clip_iters;
        median_method = cfg.median_method;
        gaussian_fitting = cfg.gaussian_fitting;
        ccl_4connectivity = cfg.ccl_4connectivity;
        source_min_area = cfg.source_min_area;
        generate_labelmap = cfg.generate_labelmap;
        generate_beam = cfg.generate_beam;
        ceres_diffmethod = cfg.ceres_diffmethod;
        ceres_solvertype = cfg.ceres_solvertype;
        find_negative_sources = cfg.find_negative_sources;
    }

    stp::SourceFindImage run_pipeline();

    // Configuration parameters
    int image_size;
    double cell_size;
    int kernel_support;
    bool kernel_exact;
    int oversampling;
    double detection_n_sigma;
    double analysis_n_sigma;
    stp::KernelFunction kernel_function;
    stp::FFTRoutine fft_routine;
    double rms_estimation;
    int sigma_clip_iters;
    stp::MedianMethod median_method;
    bool gaussian_fitting;
    bool ccl_4connectivity;
    int source_min_area;
    bool generate_labelmap;
    bool generate_beam;
    stp::CeresDiffMethod ceres_diffmethod;
    stp::CeresSolverType ceres_solvertype;
    bool find_negative_sources;
};

stp::SourceFindImage PipelineGaussianSincTest::run_pipeline()
{
    // Load simulated data from input_npz
    arma::mat input_uvw = load_npy_double_array(data_path + input_npz, "uvw_lambda");
    arma::cx_mat input_vis = load_npy_complex_array(data_path + input_npz, "vis");
    arma::mat input_snr_weights = load_npy_double_array(data_path + input_npz, "snr_weights");
    arma::mat skymodel = load_npy_double_array(data_path + input_npz, "skymodel");

    // Generate model visibilities from the skymodel and UVW-baselines
    arma::cx_mat model_vis = stp::generate_visibilities_from_local_skymodel(skymodel, input_uvw);

    // Subtract model-generated visibilities from incoming data
    arma::cx_mat residual_vis = input_vis - model_vis;

    stp::GaussianSinc kernel_func(kernel_support);
    std::pair<arma::Mat<real_t>, arma::Mat<real_t>> result = stp::image_visibilities(kernel_func, residual_vis,
        input_snr_weights, input_uvw, image_size, cell_size, kernel_support, kernel_exact, oversampling,
        generate_beam, fft_routine);

    return stp::SourceFindImage(result.first, detection_n_sigma, analysis_n_sigma, rms_estimation, find_negative_sources,
        sigma_clip_iters, median_method, gaussian_fitting, ccl_4connectivity, generate_labelmap, source_min_area,
        ceres_diffmethod, ceres_solvertype);
}

TEST_F(PipelineGaussianSincTest, test_gaussiansinc_exact)
{
    // Read config file
    std::string configfile("fastimg_exact_config.json");
    SetUp(config_path + configfile);

    // Run pipeline based on the loaded configurations
    stp::SourceFindImage sfimage = run_pipeline();

    int total_islands = sfimage.islands.size();
    EXPECT_EQ(total_islands, 1);

    int label_idx = sfimage.islands[0].label_idx;
    int sign = sfimage.islands[0].sign;
    double extremum_val = sfimage.islands[0].extremum_val;
    int extremum_x_idx = sfimage.islands[0].extremum_x_idx;
    int extremum_y_idx = sfimage.islands[0].extremum_y_idx;
    double xbar = sfimage.islands[0].moments_fit.x_centre;
    double ybar = sfimage.islands[0].moments_fit.y_centre;

    EXPECT_EQ(label_idx, 1);
    EXPECT_EQ(sign, 1);
    EXPECT_NEAR(extremum_val, 0.11045229607808273, pipeline_tolerance);
    EXPECT_EQ(extremum_x_idx, 824);
    EXPECT_EQ(extremum_y_idx, 872);
    EXPECT_NEAR(xbar, 823.43531085768677, pipeline_tolerance);
    EXPECT_NEAR(ybar, 871.6159738346181, pipeline_tolerance);
}

TEST_F(PipelineGaussianSincTest, test_gaussiansinc_oversampling)
{
    // Read config file
    std::string configfile("fastimg_oversampling_config.json");
    SetUp(config_path + configfile);

    // Run pipeline based on the loaded configurations
    stp::SourceFindImage sfimage = run_pipeline();

    int total_islands = sfimage.islands.size();
    EXPECT_EQ(total_islands, 1);

    int label_idx = sfimage.islands[0].label_idx;
    int sign = sfimage.islands[0].sign;
    double extremum_val = sfimage.islands[0].extremum_val;
    int extremum_x_idx = sfimage.islands[0].extremum_x_idx;
    int extremum_y_idx = sfimage.islands[0].extremum_y_idx;
    double xbar = sfimage.islands[0].moments_fit.x_centre;
    double ybar = sfimage.islands[0].moments_fit.y_centre;

    EXPECT_EQ(label_idx, 1);
    EXPECT_EQ(sign, 1);
    EXPECT_NEAR(extremum_val, 0.11011325990132108, pipeline_tolerance);
    EXPECT_EQ(extremum_x_idx, 824);
    EXPECT_EQ(extremum_y_idx, 872);
    EXPECT_NEAR(xbar, 823.47443901655856, pipeline_tolerance);
    EXPECT_NEAR(ybar, 871.59409370275898, pipeline_tolerance);
}
