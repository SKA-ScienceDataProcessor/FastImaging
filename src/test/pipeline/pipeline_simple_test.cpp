#include "pipeline_handler.h"
#include <gtest/gtest.h>

std::string config_path(_PIPELINE_CONFIGPATH);
std::string data_path(_PIPELINE_DATAPATH);
std::string input_npz("simdata_nstep10.npz");

// As the results are compared with python output (which uses double type) the float case presents larger error
#ifdef USE_FLOAT
const double pipeline_tolerance(1.0e-7);
#else
const double pipeline_tolerance(1.0e-11);
#endif

class SimplePipelineTest : public SlowTransientPipeline,
                           public ::testing::Test {
protected:
    void SetUp() override
    {
        // Load test-data (do it here because it is common to all tests)
        std::string datafile(data_path + input_npz);
        load_testdata(datafile);
    }
};

TEST_F(SimplePipelineTest, test_simple_gaussiansinc_exact)
{
    // Config file
    std::string configfile(config_path + "fastimg_exact_gaussiansinc_config.json");
    load_configdata(configfile);

    // Execute pipeline based on the loaded configurations
    stp::SourceFindImage sfimage = execute_pipeline();

    int total_islands = sfimage.islands.size();
    ASSERT_EQ(total_islands, 1);

    int label_idx = sfimage.islands[0].label_idx;
    int sign = sfimage.islands[0].sign;
    double extremum_val = sfimage.islands[0].extremum_val;
    int extremum_x_idx = sfimage.islands[0].extremum_x_idx;
    int extremum_y_idx = sfimage.islands[0].extremum_y_idx;
    double xbar = sfimage.islands[0].moments_fit.x_centre;
    double ybar = sfimage.islands[0].moments_fit.y_centre;

    EXPECT_EQ(label_idx, 1);
    EXPECT_EQ(sign, 1);
    EXPECT_NEAR(extremum_val, 0.11045229607808271, pipeline_tolerance);
    EXPECT_EQ(extremum_x_idx, 824);
    EXPECT_EQ(extremum_y_idx, 872);
    EXPECT_NEAR(xbar, 823.435310857686, pipeline_tolerance);
    EXPECT_NEAR(ybar, 871.615973834618, pipeline_tolerance);
}

TEST_F(SimplePipelineTest, test_simple_gaussiansinc_oversampling)
{
    // Config file
    std::string configfile(config_path + "fastimg_oversampling_gaussiansinc_config.json");
    load_configdata(configfile);

    // Execute pipeline based on the loaded configurations
    stp::SourceFindImage sfimage = execute_pipeline();

    int total_islands = sfimage.islands.size();
    ASSERT_EQ(total_islands, 1);

    int label_idx = sfimage.islands[0].label_idx;
    int sign = sfimage.islands[0].sign;
    double extremum_val = sfimage.islands[0].extremum_val;
    int extremum_x_idx = sfimage.islands[0].extremum_x_idx;
    int extremum_y_idx = sfimage.islands[0].extremum_y_idx;
    double xbar = sfimage.islands[0].moments_fit.x_centre;
    double ybar = sfimage.islands[0].moments_fit.y_centre;

    EXPECT_EQ(label_idx, 1);
    EXPECT_EQ(sign, 1);
    EXPECT_NEAR(extremum_val, 0.10973023095014588, pipeline_tolerance);
    EXPECT_EQ(extremum_x_idx, 824);
    EXPECT_EQ(extremum_y_idx, 872);
    EXPECT_NEAR(xbar, 823.363524400699, pipeline_tolerance);
    EXPECT_NEAR(ybar, 871.582414171632, pipeline_tolerance);
}

TEST_F(SimplePipelineTest, test_simple_pswf_exact)
{
    // Config file
    std::string configfile(config_path + "fastimg_exact_pswf_config.json");
    load_configdata(configfile);

    // Execute pipeline based on the loaded configurations
    stp::SourceFindImage sfimage = execute_pipeline();

    int total_islands = sfimage.islands.size();
    ASSERT_EQ(total_islands, 1);

    int label_idx = sfimage.islands[0].label_idx;
    int sign = sfimage.islands[0].sign;
    double extremum_val = sfimage.islands[0].extremum_val;
    int extremum_x_idx = sfimage.islands[0].extremum_x_idx;
    int extremum_y_idx = sfimage.islands[0].extremum_y_idx;

    EXPECT_EQ(label_idx, 1);
    EXPECT_EQ(sign, 1);
    EXPECT_NEAR(extremum_val, 0.49762858232302537, pipeline_tolerance);
    EXPECT_EQ(extremum_x_idx, 824);
    EXPECT_EQ(extremum_y_idx, 872);

    // Do not compare x_centre and y_centre (moments-fit) because there are differences
    // on the number of samples of the island. The difference seems to be caused by
    // numeric errors on sigma clip function (the function is not numerically stable).
}

TEST_F(SimplePipelineTest, test_simple_pswf_oversampling)
{
    // Config file
    std::string configfile(config_path + "fastimg_oversampling_pswf_config.json");
    load_configdata(configfile);

    // Execute pipeline based on the loaded configurations
    stp::SourceFindImage sfimage = execute_pipeline();

    int total_islands = sfimage.islands.size();
    ASSERT_EQ(total_islands, 1);

    int label_idx = sfimage.islands[0].label_idx;
    int sign = sfimage.islands[0].sign;
    double extremum_val = sfimage.islands[0].extremum_val;
    int extremum_x_idx = sfimage.islands[0].extremum_x_idx;
    int extremum_y_idx = sfimage.islands[0].extremum_y_idx;

    EXPECT_EQ(label_idx, 1);
    EXPECT_EQ(sign, 1);
    EXPECT_NEAR(extremum_val, 0.4948273376734152, pipeline_tolerance);
    EXPECT_EQ(extremum_x_idx, 824);
    EXPECT_EQ(extremum_y_idx, 872);

    // Do not compare x_centre and y_centre (moments-fit) because there are differences
    // on the number of samples of the island. The difference seems to be caused by
    // numeric errors on sigma clip function (the function is not numerically stable).
}
