/** @file sourcefind_test_Fitting.cpp
 *  @brief Test gaussian fitting function
 */

#include <fixtures.h>
#include <gaussian2d.h>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

const double fit_tolerance(1.0e-5);

class SourceFindFitting : public ::testing::TestWithParam<testing::tuple<CeresDiffMethod, CeresSolverType>> {
private:
    double ydim;
    double xdim;
    double rms;

    double detection_n_sigma;
    double analysis_n_sigma;
    double rms_est;
    bool find_negative_sources;
    uint sigma_clip_iters;
    bool binapprox_median;
    bool gaussian_fitting;
    bool generate_labelmap;
    CeresDiffMethod ceres_diffmethod;
    CeresSolverType ceres_solvertype;

    double bright_x_centre;
    double bright_y_centre;
    double bright_amplitude;

    double source_x_stddev;
    double source_y_stddev;
    double source_theta;

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
        sigma_clip_iters = 5;
        binapprox_median = false;
        gaussian_fitting = true;
        generate_labelmap = true;
        ceres_diffmethod = ::testing::get<0>(GetParam());
        ceres_solvertype = ::testing::get<1>(GetParam());

        bright_x_centre = 48.24;
        bright_y_centre = 52.66;
        bright_amplitude = 10.0;

        source_x_stddev = 1.5;
        source_y_stddev = 1.2;
        source_theta = 0.5;

        bright_src = gaussian_point_source(bright_x_centre, bright_y_centre, bright_amplitude, source_x_stddev, source_y_stddev, source_theta);
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
        SourceFindImage sf(img, detection_n_sigma, analysis_n_sigma, rms_est, find_negative_sources,
            sigma_clip_iters, binapprox_median, gaussian_fitting, generate_labelmap, ceres_diffmethod, ceres_solvertype);
#ifndef FFTSHIFT
        fftshift(img);
#endif
        found_src = sf.islands[0];

        total_islands = sf.islands.size();

        absolute_x_idx = std::abs(found_src.extremum_x_idx - bright_src.x_mean);
        absolute_y_idx = std::abs(found_src.extremum_y_idx - bright_src.y_mean);
    }

    Gaussian2D bright_src;

    double total_islands;
    double absolute_x_idx;
    double absolute_y_idx;

    IslandParams found_src;
};

TEST_P(SourceFindFitting, GaussianFitting)
{
    EXPECT_EQ(total_islands, 1);
    EXPECT_LT(absolute_x_idx, 0.5);
    EXPECT_LT(absolute_y_idx, 0.5);
    EXPECT_NEAR(found_src.leastsq_fit.amplitude, bright_src.amplitude, fit_tolerance);
    EXPECT_NEAR(found_src.leastsq_fit.x_centre, bright_src.x_mean, fit_tolerance);
    EXPECT_NEAR(found_src.leastsq_fit.y_centre, bright_src.y_mean, fit_tolerance);
    EXPECT_NEAR(found_src.leastsq_fit.semimajor, bright_src.x_stddev, fit_tolerance);
    EXPECT_NEAR(found_src.leastsq_fit.semiminor, bright_src.y_stddev, fit_tolerance);
    EXPECT_NEAR(found_src.leastsq_fit.theta, bright_src.theta, fit_tolerance);
}

INSTANTIATE_TEST_CASE_P(DifferentCeresParameters,
    SourceFindFitting,
    ::testing::Combine(::testing::Values(CeresDiffMethod::AutoDiff_SingleResBlk, CeresDiffMethod::AutoDiff, CeresDiffMethod::AnalyticDiff_SingleResBlk, CeresDiffMethod::AnalyticDiff),
        ::testing::Values(CeresSolverType::LinearSearch_BFGS, CeresSolverType::LinearSearch_LBFGS, CeresSolverType::TrustRegion_DenseQR)));
