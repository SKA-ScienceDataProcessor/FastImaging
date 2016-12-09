#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <libstp.h>

class SourceFindRmsEstimation : public ::testing::Test {
private:
    double ydim;
    double xdim;
    double rms;
    double rms_est;

    double bright_x_centre;
    double bright_y_centre;
    double bright_amplitude;

    double faint_x_centre;
    double faint_y_centre;
    double faint_amplitude;

public:
    void SetUp()
    {
        ydim = 128;
        xdim = 64;
        rms = 1.0;

        bright_x_centre = 48.24;
        bright_y_centre = 52.66;
        bright_amplitude = 10.0;

        faint_x_centre = 32;
        faint_y_centre = 64;
        faint_amplitude = 3.5;
    }

    void run()
    {
        arma::mat img = uncorrelated_gaussian_noise_background(ydim, xdim, rms);
        img += evaluate_model_on_pixel_grid(ydim, xdim, gaussian_point_source(bright_x_centre, bright_y_centre, bright_amplitude));
        img += evaluate_model_on_pixel_grid(ydim, xdim, gaussian_point_source(faint_x_centre, faint_y_centre, faint_amplitude));

        rms_est = _estimate_rms(img);

        arma::mat aux = { (rms_est - rms) / rms };
        absolute_rms = arma::abs(aux)[0];
    }
    double absolute_rms;
};

TEST_F(SourceFindRmsEstimation, Absolute_rms)
{
    run();
    EXPECT_LT(absolute_rms, 0.05);
}

TEST_F(SourceFindRmsEstimation, SourceFindRmsEstimation_benchmark)
{
    benchmark::RegisterBenchmark("SourceFindRmsEstimation", [this](benchmark::State& state) { while(state.KeepRunning())run(); });
    benchmark::RunSpecifiedBenchmarks();
}
