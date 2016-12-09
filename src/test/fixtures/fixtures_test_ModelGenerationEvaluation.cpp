#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <libstp.h>

class FixturesModelGenerationEvaluation : public ::testing::Test {
private:
    double ydim;
    double xdim;
    double gaussian2d_x_mean;
    double gaussian2d_y_mean;
    Gaussian2D src;

public:
    void SetUp()
    {
        ydim = 128;
        xdim = 64;
        gaussian2d_x_mean = 32;
        gaussian2d_y_mean = 64;

        img = arma::zeros(ydim, xdim);
        src = gaussian_point_source(gaussian2d_x_mean, gaussian2d_y_mean);
    }

    void run()
    {
        img += evaluate_model_on_pixel_grid(ydim, xdim, src);
        arma::mat aux = { img.at(src.y_mean, src.x_mean) - src.amplitude };
        absolute_rms = arma::abs(aux)[0];
    }

    arma::mat img;
    double absolute_rms;
};

TEST_F(FixturesModelGenerationEvaluation, test0)
{
    run();
    EXPECT_EQ(img.at(0, 0), 0.0);
}

TEST_F(FixturesModelGenerationEvaluation, test1)
{
    run();
    EXPECT_LT(absolute_rms, 0.01);
}

TEST_F(FixturesModelGenerationEvaluation, FixturesModelGenerationEvaluation_benchmark)
{
    benchmark::RegisterBenchmark("FixturesModelGenerationEvaluation", [this](benchmark::State& state) { while(state.KeepRunning())run(); });
    benchmark::RunSpecifiedBenchmarks();
}
