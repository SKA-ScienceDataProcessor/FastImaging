/** @file sourcefind_test_NegativeSourceDetection.cpp
 *  @brief Test SourceFindImage module implementation for a negative source detection
 */

#include <fixtures.h>
#include <gaussian2d.h>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class SourceFindNegativeSourceDetection : public ::testing::Test {
private:
    double ydim;
    double xdim;
    double rms;

    double detection_n_sigma;
    double analysis_n_sigma;
    double rms_est;
    bool find_negative_sources;

    double bright_x_centre;
    double bright_y_centre;
    double bright_amplitude;

    double faint_x_centre;
    double faint_y_centre;
    double faint_amplitude;

    double negative_x_centre;
    double negative_y_centre;
    double negative_amplitude;

    double source_x_stddev;
    double source_y_stddev;
    double source_theta;

    IslandParams found_src;

    arma::Mat<real_t> img;

public:
    void SetUp()
    {
        ydim = 128;
        xdim = 64;
        rms = 1.0;

        detection_n_sigma = 4;
        analysis_n_sigma = 3;
        rms_est = rms;
        find_negative_sources = true;

        bright_x_centre = 48.24;
        bright_y_centre = 52.66;
        bright_amplitude = 10.0;

        faint_x_centre = 32;
        faint_y_centre = 64;
        faint_amplitude = 3.5;

        negative_x_centre = 24.31;
        negative_y_centre = 28.157;
        negative_amplitude = -10.0;

        source_x_stddev = 1.5;
        source_y_stddev = 1.2;
        source_theta = 1.0;

        bright_src = gaussian_point_source(bright_x_centre, bright_y_centre, bright_amplitude, source_x_stddev, source_y_stddev, source_theta);
        faint_src = gaussian_point_source(faint_x_centre, faint_y_centre, faint_amplitude, source_x_stddev, source_y_stddev, source_theta);
        negative_src = gaussian_point_source(negative_x_centre, negative_y_centre, negative_amplitude, source_x_stddev, source_y_stddev, source_theta);

        img = arma::zeros<arma::Mat<real_t>>(ydim, xdim);

        // Run source find
        run();
    }

    void run()
    {
        img += evaluate_model_on_pixel_grid(ydim, xdim, negative_src);
#ifndef FFTSHIFT
        // Input data needs to be shifted because source_find assumes it is shifted (required when FFTSHIFT option is disabled)
        fftshift(img);
#endif
        SourceFindImage sf(img, detection_n_sigma, analysis_n_sigma, rms_est, find_negative_sources);
#ifndef FFTSHIFT
        fftshift(img);
#endif
        found_src = sf.islands[0];

        total_islands0 = sf.islands.size();

        absolute_x_idx = std::abs(found_src.extremum_x_idx - negative_src.x_mean);
        absolute_y_idx = std::abs(found_src.extremum_y_idx - negative_src.y_mean);
        absolute_xbar = std::abs(found_src.xbar - negative_src.x_mean);
        absolute_ybar = std::abs(found_src.ybar - negative_src.y_mean);

        img += evaluate_model_on_pixel_grid(ydim, xdim, bright_src);
#ifndef FFTSHIFT
        fftshift(img);
#endif
        sf = SourceFindImage(img, detection_n_sigma, analysis_n_sigma, rms_est, find_negative_sources);
#ifndef FFTSHIFT
        fftshift(img);
#endif
        total_islands1 = sf.islands.size();

        for (arma::uword l_idx(0); l_idx < sf.islands.size(); ++l_idx) {
            if (sf.islands[l_idx].sign == 1) {
                positive_islands.push_back(sf.islands[l_idx]);
            } else {
                negative_islands.push_back(sf.islands[l_idx]);
            }
        }
        total_positive_islands = positive_islands.size();
        total_negative_islands = negative_islands.size();

        same_island = (found_src == negative_islands[0]);
    }

    Gaussian2D bright_src;
    Gaussian2D faint_src;
    Gaussian2D negative_src;

    std::vector<IslandParams> positive_islands;
    std::vector<IslandParams> negative_islands;

    double total_islands0;
    double total_islands1;

    double absolute_x_idx;
    double absolute_y_idx;
    double absolute_xbar;
    double absolute_ybar;

    double total_positive_islands;
    double total_negative_islands;

    bool same_island;
};

TEST_F(SourceFindNegativeSourceDetection, Total_islands)
{
    EXPECT_EQ(total_islands0, 1);
    EXPECT_EQ(total_islands1, 2);
    EXPECT_EQ(total_positive_islands, 1);
    EXPECT_EQ(total_negative_islands, 1);
}

TEST_F(SourceFindNegativeSourceDetection, Negative_island)
{
    EXPECT_LT(absolute_x_idx, 0.5);
    EXPECT_LT(absolute_y_idx, 0.5);
    EXPECT_LT(absolute_xbar, 0.1);
    EXPECT_LT(absolute_ybar, 0.1);
    EXPECT_TRUE(same_island);
}

TEST_F(SourceFindNegativeSourceDetection, Positive_island)
{
    EXPECT_LT(positive_islands[0].extremum_x_idx - bright_src.x_mean, 0.5);
    EXPECT_LT(positive_islands[0].xbar - bright_src.x_mean, 0.1);
    EXPECT_LT(positive_islands[0].extremum_y_idx - bright_src.y_mean, 0.5);
    EXPECT_LT(positive_islands[0].ybar - bright_src.y_mean, 0.1);
}
