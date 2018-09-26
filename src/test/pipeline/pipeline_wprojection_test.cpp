#include "pipeline_handler.h"
#include <gtest/gtest.h>

std::string config_path(_PIPELINE_CONFIGPATH);
std::string data_path(_PIPELINE_DATAPATH);
std::string wproj_input_npz("simdata_awproj_nstep10.npz");

// As the results are compared with python output (which uses double type) the float case presents larger error
#ifdef USE_FLOAT
const double pipeline_tolerance(1.0e-5);
#else
const double pipeline_tolerance(1.0e-8);
#endif

class WprojectionPipelineTest : public SlowTransientPipeline,
                                public ::testing::Test {
protected:
    void SetUp() override
    {
        // Load test-data (do it here because it is common to all tests)
        std::string datafile(data_path + wproj_input_npz);
        load_testdata(datafile);
    }
};

#ifdef WPROJECTION
TEST_F(WprojectionPipelineTest, test_wproj_mean)
{
    // Config file
    std::string configfile(config_path + "fastimg_wproj_mean_config.json");
    load_configdata(configfile);

    // Execute pipeline based on the loaded configurations
    stp::SourceFindImage sfimage = execute_pipeline();

    // Expected results
    int expected_total_isl = 5;
    std::vector<double> expected_extremum_value = { 0.9912493658247081, 0.9806111166040254, 0.9611388809803832, 0.9308721168353326, 0.9309723162928002 };
    std::vector<double> expected_extremum_x = { 512, 631, 713, 786, 851 };
    std::vector<double> expected_extremum_y = { 512, 658, 770, 882, 995 };

    // Check results
    int total_islands = sfimage.islands.size();
    ASSERT_EQ(total_islands, expected_total_isl);

    for (int idx = 0; idx < expected_total_isl; ++idx) {
        EXPECT_EQ(sfimage.islands[idx].sign, 1);
        EXPECT_NEAR(sfimage.islands[idx].extremum_val, expected_extremum_value[idx], pipeline_tolerance);
        EXPECT_EQ(sfimage.islands[idx].extremum_x_idx, expected_extremum_x[idx]);
        EXPECT_EQ(sfimage.islands[idx].extremum_y_idx, expected_extremum_y[idx]);
        idx++;
    }
}

TEST_F(WprojectionPipelineTest, test_wproj_median)
{
    // Config file
    std::string configfile(config_path + "fastimg_wproj_median_config.json");
    load_configdata(configfile);

    // Execute pipeline based on the loaded configurations
    stp::SourceFindImage sfimage = execute_pipeline();

    // Expected results
    int expected_total_isl = 5;
    std::vector<double> expected_extremum_value = { 0.9912487225137617, 0.9813749725747497, 0.960673810727429, 0.9312208247213017, 0.929783156438092 };
    std::vector<double> expected_extremum_x = { 512, 631, 713, 786, 851 };
    std::vector<double> expected_extremum_y = { 512, 658, 770, 882, 995 };

    // Check results
    int total_islands = sfimage.islands.size();
    ASSERT_EQ(total_islands, expected_total_isl);

    for (int idx = 0; idx < expected_total_isl; ++idx) {
        EXPECT_EQ(sfimage.islands[idx].sign, 1);
        EXPECT_NEAR(sfimage.islands[idx].extremum_val, expected_extremum_value[idx], pipeline_tolerance);
        EXPECT_EQ(sfimage.islands[idx].extremum_x_idx, expected_extremum_x[idx]);
        EXPECT_EQ(sfimage.islands[idx].extremum_y_idx, expected_extremum_y[idx]);
        idx++;
    }
}

TEST_F(WprojectionPipelineTest, test_wproj_hankel)
{
    // Config file
    std::string configfile(config_path + "fastimg_wproj_hankel_config.json");
    load_configdata(configfile);

    // Execute pipeline based on the loaded configurations
    stp::SourceFindImage sfimage = execute_pipeline();

    // Expected results
    int expected_total_isl = 5;
    std::vector<double> expected_extremum_value = { 0.99126548796787806, 0.97916638485134411, 0.96251544620881657, 0.96036208179378824, 1.2258315512118649 };
    std::vector<double> expected_extremum_x = { 512, 631, 713, 786, 851 };
    std::vector<double> expected_extremum_y = { 512, 658, 770, 882, 995 };

    // Check results
    int total_islands = sfimage.islands.size();
    ASSERT_EQ(total_islands, expected_total_isl);

    for (int idx = 0; idx < expected_total_isl; ++idx) {
        EXPECT_EQ(sfimage.islands[idx].sign, 1);
        EXPECT_NEAR(sfimage.islands[idx].extremum_val, expected_extremum_value[idx], pipeline_tolerance);
        EXPECT_EQ(sfimage.islands[idx].extremum_x_idx, expected_extremum_x[idx]);
        EXPECT_EQ(sfimage.islands[idx].extremum_y_idx, expected_extremum_y[idx]);
        idx++;
    }
}
#endif
