#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <libstp.h>

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

    island_params found_src;

    Gaussian2D bright_src;
    Gaussian2D faint_src;
    Gaussian2D negative_src;

    source_find_image sf;

    arma::mat img;

    std::vector<island_params> positive_islands;
    std::vector<island_params> negative_islands;

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

        bright_src = gaussian_point_source(bright_x_centre, bright_y_centre, bright_amplitude);
        faint_src = gaussian_point_source(faint_x_centre, faint_y_centre, faint_amplitude);
        negative_src = gaussian_point_source(negative_x_centre, negative_y_centre, negative_amplitude);

        img = arma::zeros(ydim, xdim);
    }

    void run()
    {
        img += evaluate_model_on_pixel_grid(ydim, xdim, negative_src);
        sf = source_find_image(img, detection_n_sigma, analysis_n_sigma, rms_est, find_negative_sources);
        found_src = sf.islands[0];

        total_islands0 = sf.islands.size();

        absolute_x_idx = abs(found_src.extremum_x_idx - negative_src.x_mean);
        absolute_y_idx = abs(found_src.extremum_y_idx - negative_src.y_mean);
        absolute_xbar = abs(found_src.xbar - negative_src.x_mean);
        absolute_ybar = abs(found_src.ybar - negative_src.y_mean);

        img += evaluate_model_on_pixel_grid(ydim, xdim, bright_src);
        sf = source_find_image(img, detection_n_sigma, analysis_n_sigma, rms_est, find_negative_sources);
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
    run();
    EXPECT_EQ(total_islands0, 1);
}

TEST_F(SourceFindNegativeSourceDetection, Absolute_X_idx)
{
    run();
    EXPECT_LT(absolute_x_idx, 0.5);
}

TEST_F(SourceFindNegativeSourceDetection, Absolute_Y_idx)
{
    run();
    EXPECT_LT(absolute_y_idx, 0.5);
}

TEST_F(SourceFindNegativeSourceDetection, Absolute_Xbar)
{
    run();
    EXPECT_LT(absolute_xbar, 0.1);
}

TEST_F(SourceFindNegativeSourceDetection, Absolute_Ybar)
{
    run();
    EXPECT_LT(absolute_ybar, 0.1);
}

TEST_F(SourceFindNegativeSourceDetection, Total_islands1)
{
    run();
    EXPECT_EQ(total_islands1, 2);
}

TEST_F(SourceFindNegativeSourceDetection, Total_positive_islands)
{
    run();
    EXPECT_EQ(total_positive_islands, 1);
}

TEST_F(SourceFindNegativeSourceDetection, Total_negative_islands)
{
    run();
    EXPECT_EQ(total_negative_islands, 1);
}

TEST_F(SourceFindNegativeSourceDetection, Negative_same_island)
{
    run();
    EXPECT_TRUE(same_island);
}

TEST_F(SourceFindNegativeSourceDetection, SourceFindNegativeSourceDetection_benchmark)
{
    benchmark::RegisterBenchmark("SourceFindNegativeSourceDetection", [this](benchmark::State& state) { while(state.KeepRunning())run(); });
    benchmark::RunSpecifiedBenchmarks();
}
