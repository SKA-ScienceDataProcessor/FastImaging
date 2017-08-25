/** @file sourcefind_testBasicSourceDetection.cpp
 *  @brief Test SourceFindImage module implementation
 *         for a basic source detection
 */

#include <fixtures.h>
#include <gaussian2d.h>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class SourceFindBasicSourceDetection : public ::testing::Test {
private:
    double ydim;
    double xdim;
    double rms;
    double rms_est;
    double detection_n_sigma;
    double analysis_n_sigma;
    double bright_x_centre;
    double bright_y_centre;
    double bright_amplitude;
    double faint_x_centre;
    double faint_y_centre;
    double faint_amplitude;
    double source_x_stddev;
    double source_y_stddev;
    double source_theta;
    bool find_negative_sources;

    IslandParams found_src;
    Gaussian2D bright_src;
    Gaussian2D faint_src;
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
        find_negative_sources = false;

        bright_x_centre = 48.24;
        bright_y_centre = 52.66;
        bright_amplitude = 10.0;

        faint_x_centre = 32;
        faint_y_centre = 64;
        faint_amplitude = 3.5;

        source_x_stddev = 1.5;
        source_y_stddev = 1.2;
        source_theta = 1.0;

        bright_src = gaussian_point_source(bright_x_centre, bright_y_centre, bright_amplitude, source_x_stddev, source_y_stddev, source_theta);
        faint_src = gaussian_point_source(faint_x_centre, faint_y_centre, faint_amplitude, source_x_stddev, source_y_stddev, source_theta);
        img = arma::zeros<arma::Mat<real_t>>(ydim, xdim);

        // Run source find
        run();
    }

    void run()
    {
        img += evaluate_model_on_pixel_grid(ydim, xdim, bright_src);
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

        absolute_x_idx = std::abs(found_src.extremum_x_idx - bright_src.x_mean);
        absolute_y_idx = std::abs(found_src.extremum_y_idx - bright_src.y_mean);
        absolute_xbar = std::abs(found_src.xbar - bright_src.x_mean);
        absolute_ybar = std::abs(found_src.ybar - bright_src.y_mean);

        img += evaluate_model_on_pixel_grid(ydim, xdim, faint_src);
#ifndef FFTSHIFT
        fftshift(img);
#endif
        sf = SourceFindImage(img, detection_n_sigma, analysis_n_sigma, rms_est, find_negative_sources);
#ifndef FFTSHIFT
        fftshift(img);
#endif
        total_islands1 = sf.islands.size();

        img += evaluate_model_on_pixel_grid(ydim, xdim, faint_src);
#ifndef FFTSHIFT
        fftshift(img);
#endif
        sf = SourceFindImage(img, detection_n_sigma, analysis_n_sigma, rms_est, find_negative_sources);
#ifndef FFTSHIFT
        fftshift(img);
#endif
        total_islands2 = sf.islands.size();
    }

    double total_islands0;
    double total_islands1;
    double total_islands2;

    double absolute_x_idx;
    double absolute_y_idx;
    double absolute_xbar;
    double absolute_ybar;
};

TEST_F(SourceFindBasicSourceDetection, Total_islands0)
{
    EXPECT_EQ(total_islands0, 1);
}

TEST_F(SourceFindBasicSourceDetection, Absolute_x_idx)
{
    EXPECT_LT(absolute_x_idx, 0.5);
}

TEST_F(SourceFindBasicSourceDetection, Absolute_y_idx)
{
    EXPECT_LT(absolute_y_idx, 0.5);
}

TEST_F(SourceFindBasicSourceDetection, Absolute_xbar)
{
    EXPECT_LT(absolute_xbar, 0.1);
}

TEST_F(SourceFindBasicSourceDetection, Absolute_ybar)
{
    EXPECT_LT(absolute_ybar, 0.1);
}

TEST_F(SourceFindBasicSourceDetection, Total_islands1)
{
    EXPECT_EQ(total_islands1, 1);
}

TEST_F(SourceFindBasicSourceDetection, Total_islands2)
{
    EXPECT_EQ(total_islands2, 2);
}
