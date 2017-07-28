/** @file sourcefind_testBasicSourceDetection.cpp
 *  @brief Test SourceFindImage module implementation
 *         for a basic source detection
 *
 *  @bug No known bugs.
 */

#include <fixtures.h>
#include <gaussian2d.h>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class SourceFindLabeling : public ::testing::Test {
private:
    double rms_est;
    double detection_n_sigma;
    double analysis_n_sigma;
    bool find_negative_sources;

    island_params found_src;
    arma::Mat<real_t> img;

public:
    void SetUp()
    {
        detection_n_sigma = 4;
        analysis_n_sigma = 3;
        rms_est = 1.0;
        find_negative_sources = false;
    }

    void run()
    {
        real_t S = 10.0; // source pixel
        real_t n = 0.0; // empty pixel

        img = {
            { S, S, n, n, n, n, S, S },
            { n, n, n, S, n, n, S, n },
            { n, n, n, n, n, n, n, n },
            { S, n, n, S, S, n, n, S },
            { S, n, n, S, S, n, n, S },
            { n, n, n, n, n, n, n, n },
            { n, n, n, n, n, n, S, n },
            { S, S, n, n, n, n, S, n },
        };

// Order of assigned labels changes with non-shifted matrices.
// Thus two results are possible depending whether the matrix is shifted or not
#ifndef FFTSHIFT
        expected_map = {
            { 3, 3, 0, 0, 0, 0, 7, 7 },
            { 0, 0, 0, 5, 0, 0, 7, 0 },
            { 0, 0, 0, 0, 0, 0, 0, 0 },
            { 1, 0, 0, 4, 4, 0, 0, 8 },
            { 1, 0, 0, 4, 4, 0, 0, 8 },
            { 0, 0, 0, 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0, 0, 6, 0 },
            { 2, 2, 0, 0, 0, 0, 6, 0 },
        };
#else
        expected_map = {
            { 1, 1, 0, 0, 0, 0, 6, 6 },
            { 0, 0, 0, 4, 0, 0, 6, 0 },
            { 0, 0, 0, 0, 0, 0, 0, 0 },
            { 2, 0, 0, 5, 5, 0, 0, 8 },
            { 2, 0, 0, 5, 5, 0, 0, 8 },
            { 0, 0, 0, 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0, 0, 7, 0 },
            { 3, 3, 0, 0, 0, 0, 7, 0 },
        };
#endif

#ifndef FFTSHIFT
        // Input data needs to be shifted because source_find assumes it is shifted (required when FFTSHIFT option is disabled)
        fftshift(img);
#endif
        source_find_image sf(img, detection_n_sigma, analysis_n_sigma, rms_est, find_negative_sources);

        total_islands = sf.islands.size();
        result_map = static_cast<arma::Mat<int>>(sf.label_map);
#ifndef FFTSHIFT
        fftshift(result_map);
#endif
    }

    int total_islands;
    int expected_total_islands = 8;
    arma::Mat<int> expected_map;
    arma::Mat<int> result_map;
};

TEST_F(SourceFindLabeling, Total_islands)
{
    run();
    EXPECT_EQ(total_islands, expected_total_islands);
}

TEST_F(SourceFindLabeling, LabelMap)
{
    run();
    EXPECT_EQ(arma::accu(result_map == expected_map), expected_map.n_elem);
}
