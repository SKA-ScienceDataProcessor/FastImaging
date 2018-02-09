/** @file sourcefind_test_Labeling.cpp
 *  @brief Test SourceFindImage labeling function
 */

#include <fixtures.h>
#include <gaussian2d.h>
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

class SourceFindLabeling : public ::testing::Test {
public:
    double rms_est;
    double detection_n_sigma;
    double analysis_n_sigma;
    bool find_negative_sources;
    uint sigma_clip_iters;
    MedianMethod median_method;
    bool gaussian_fitting;
    bool ccl_4connectivity;
    bool generate_labelmap;
    int source_min_area;
    IslandParams found_src;

    int total_positive_islands;
    int total_negative_islands;
    int expected_total_islands;
    arma::Mat<real_t> img;
    arma::Mat<int> expected_map;
    arma::Mat<int> result_positive_map;
    arma::Mat<int> result_negative_map;

    void SetUp()
    {
        detection_n_sigma = 4;
        analysis_n_sigma = 3;
        rms_est = 1.0;
        find_negative_sources = true;
        sigma_clip_iters = 5;
        median_method = MedianMethod::ZEROMEDIAN;
        gaussian_fitting = false;
        ccl_4connectivity = false;
        generate_labelmap = false;
        source_min_area = 1;
    }

    void run()
    {
#ifndef FFTSHIFT
        // Input data needs to be shifted because source_find assumes it is shifted (required when FFTSHIFT option is disabled)
        fftshift(img);
#endif
        SourceFindImage sf(img, detection_n_sigma, analysis_n_sigma, rms_est, find_negative_sources, sigma_clip_iters,
            median_method, gaussian_fitting, ccl_4connectivity, generate_labelmap, source_min_area);
        total_positive_islands = sf.islands.size();
        result_positive_map = static_cast<arma::Mat<int>>(sf.label_map);
#ifndef FFTSHIFT
        fftshift(result_positive_map);
#endif
        // Invert image values and perform new labeling
        img *= (-1);
        SourceFindImage sfn(img, detection_n_sigma, analysis_n_sigma, rms_est, find_negative_sources, sigma_clip_iters,
            median_method, gaussian_fitting, ccl_4connectivity, generate_labelmap, source_min_area);
        total_negative_islands = sfn.islands.size();
        result_negative_map = static_cast<arma::Mat<int>>(sfn.label_map);
#ifndef FFTSHIFT
        fftshift(result_negative_map);
#endif
    }
};

TEST_F(SourceFindLabeling, 4CCL_ImageRand1)
{
    real_t S = 10.0; // source pixel
    real_t n = 0.0; // empty pixel

    ccl_4connectivity = true;
    img = {
        { S, S, n, n, n, n, S, S },
        { n, n, n, S, n, S, n, n },
        { n, n, n, n, n, n, n, n },
        { S, n, n, S, S, n, n, S },
        { S, n, n, S, S, n, n, S },
        { n, n, S, n, n, S, n, n },
        { n, n, n, n, n, n, S, n },
        { S, S, n, n, n, n, S, n },
    };
    run();

// Order of assigned labels changes with non-shifted matrices.
// Thus two results are possible depending whether the matrix is shifted or not
#ifndef FFTSHIFT
    expected_map = {
        { 3, 3, 0, 0, 0, 0, 10, 10 },
        { 0, 0, 0, 6, 0, 8, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 0, 0, 5, 5, 0, 0, 11 },
        { 1, 0, 0, 5, 5, 0, 0, 11 },
        { 0, 0, 4, 0, 0, 7, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 9, 0 },
        { 2, 2, 0, 0, 0, 0, 9, 0 },
    };
#else
    expected_map = {
        { 1, 1, 0, 0, 0, 0, 9, 9 },
        { 0, 0, 0, 5, 0, 7, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, 0, 0, 6, 6, 0, 0, 11 },
        { 2, 0, 0, 6, 6, 0, 0, 11 },
        { 0, 0, 4, 0, 0, 8, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 10, 0 },
        { 3, 3, 0, 0, 0, 0, 10, 0 },
    };
#endif
    expected_total_islands = 11;

    EXPECT_EQ(total_positive_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_positive_map == expected_map), expected_map.n_elem);

    EXPECT_EQ(total_negative_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_negative_map == (expected_map * (-1))), expected_map.n_elem);
}

TEST_F(SourceFindLabeling, 8CCL_ImageRand1)
{
    real_t S = 10.0; // source pixel
    real_t n = 0.0; // empty pixel

    ccl_4connectivity = false;
    img = {
        { S, S, n, n, n, n, S, S },
        { n, n, n, S, n, S, n, n },
        { n, n, n, n, n, n, n, n },
        { S, n, n, S, S, n, n, S },
        { S, n, n, S, S, n, n, S },
        { n, n, S, n, n, S, n, n },
        { n, n, n, n, n, n, S, n },
        { S, S, n, n, n, n, S, n },
    };
    run();

#ifndef FFTSHIFT
    expected_map = {
        { 3, 3, 0, 0, 0, 0, 6, 6 },
        { 0, 0, 0, 5, 0, 6, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 0, 0, 4, 4, 0, 0, 7 },
        { 1, 0, 0, 4, 4, 0, 0, 7 },
        { 0, 0, 4, 0, 0, 4, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 4, 0 },
        { 2, 2, 0, 0, 0, 0, 4, 0 },
    };
#else
    expected_map = {
        { 1, 1, 0, 0, 0, 0, 6, 6 },
        { 0, 0, 0, 5, 0, 6, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, 0, 0, 4, 4, 0, 0, 7 },
        { 2, 0, 0, 4, 4, 0, 0, 7 },
        { 0, 0, 4, 0, 0, 4, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 4, 0 },
        { 3, 3, 0, 0, 0, 0, 4, 0 },
    };
#endif
    expected_total_islands = 7;

    EXPECT_EQ(total_positive_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_positive_map == expected_map), expected_map.n_elem);

    EXPECT_EQ(total_negative_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_negative_map == (expected_map * (-1))), expected_map.n_elem);
}

TEST_F(SourceFindLabeling, 4CCL_ImageBorderObj)
{
    real_t S = 10.0; // source pixel
    real_t n = 0.0; // empty pixel

    ccl_4connectivity = true;
    img = {
        { S, S, S, S, S, S, S, S },
        { S, n, n, n, n, n, n, S },
        { S, n, n, n, n, n, n, S },
        { S, n, n, S, S, n, n, S },
        { S, n, n, S, S, n, n, S },
        { S, n, n, n, n, n, n, S },
        { S, n, n, n, n, n, n, S },
        { S, S, S, S, S, S, S, S },
    };
    run();

#ifndef FFTSHIFT
    expected_map = {
        { 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 2, 2, 0, 0, 1 },
        { 1, 0, 0, 2, 2, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1 },
    };
#else
    expected_map = {
        { 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 2, 2, 0, 0, 1 },
        { 1, 0, 0, 2, 2, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1 },
    };
#endif
    expected_total_islands = 2;

    EXPECT_EQ(total_positive_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_positive_map == expected_map), expected_map.n_elem);

    EXPECT_EQ(total_negative_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_negative_map == (expected_map * (-1))), expected_map.n_elem);
}

TEST_F(SourceFindLabeling, 8CCL_ImageBorderObj)
{
    real_t S = 10.0; // source pixel
    real_t n = 0.0; // empty pixel

    ccl_4connectivity = false;
    img = {
        { S, S, S, S, S, S, S, S },
        { S, n, n, n, n, n, n, S },
        { S, n, n, n, n, n, n, S },
        { S, n, n, S, S, n, n, S },
        { S, n, n, S, S, n, n, S },
        { S, n, n, n, n, n, n, S },
        { S, n, n, n, n, n, n, S },
        { S, S, S, S, S, S, S, S },
    };
    run();

#ifndef FFTSHIFT
    expected_map = {
        { 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 2, 2, 0, 0, 1 },
        { 1, 0, 0, 2, 2, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1 },
    };
#else
    expected_map = {
        { 1, 1, 1, 1, 1, 1, 1, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 2, 2, 0, 0, 1 },
        { 1, 0, 0, 2, 2, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 0, 0, 0, 0, 0, 0, 1 },
        { 1, 1, 1, 1, 1, 1, 1, 1 },
    };
#endif
    expected_total_islands = 2;

    EXPECT_EQ(total_positive_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_positive_map == expected_map), expected_map.n_elem);

    EXPECT_EQ(total_negative_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_negative_map == (expected_map * (-1))), expected_map.n_elem);
}

TEST_F(SourceFindLabeling, 4CCL_ImageCornerObjs)
{
    real_t S = 10.0; // source pixel
    real_t n = 0.0; // empty pixel

    ccl_4connectivity = true;
    img = {
        { S, n, n, n, n, n, n, S },
        { n, n, n, n, n, n, n, n },
        { n, n, n, n, n, n, n, n },
        { n, n, n, n, S, n, n, n },
        { n, n, n, S, n, n, n, n },
        { n, n, n, n, n, n, n, n },
        { n, n, n, n, n, n, n, n },
        { S, n, n, n, n, n, n, S },
    };
    run();

#ifndef FFTSHIFT
    expected_map = {
        { 2, 0, 0, 0, 0, 0, 0, 6 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 4, 0, 0, 0 },
        { 0, 0, 0, 3, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 0, 0, 0, 0, 0, 0, 5 },
    };
#else
    expected_map = {
        { 1, 0, 0, 0, 0, 0, 0, 5 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 4, 0, 0, 0 },
        { 0, 0, 0, 3, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, 0, 0, 0, 0, 0, 0, 6 },
    };
#endif
    expected_total_islands = 6;

    EXPECT_EQ(total_positive_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_positive_map == expected_map), expected_map.n_elem);

    EXPECT_EQ(total_negative_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_negative_map == (expected_map * (-1))), expected_map.n_elem);
}

TEST_F(SourceFindLabeling, 8CCL_ImageCornerObjs)
{
    real_t S = 10.0; // source pixel
    real_t n = 0.0; // empty pixel

    ccl_4connectivity = false;
    img = {
        { S, n, n, n, n, n, n, S },
        { n, n, n, n, n, n, n, n },
        { n, n, n, n, n, n, n, n },
        { n, n, n, n, S, n, n, n },
        { n, n, n, S, n, n, n, n },
        { n, n, n, n, n, n, n, n },
        { n, n, n, n, n, n, n, n },
        { S, n, n, n, n, n, n, S },
    };
    run();

#ifndef FFTSHIFT
    expected_map = {
        { 2, 0, 0, 0, 0, 0, 0, 5 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 3, 0, 0, 0 },
        { 0, 0, 0, 3, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 1, 0, 0, 0, 0, 0, 0, 4 },
    };
#else
    expected_map = {
        { 1, 0, 0, 0, 0, 0, 0, 4 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 3, 0, 0, 0 },
        { 0, 0, 0, 3, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 2, 0, 0, 0, 0, 0, 0, 5 },
    };
#endif
    expected_total_islands = 5;

    EXPECT_EQ(total_positive_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_positive_map == expected_map), expected_map.n_elem);

    EXPECT_EQ(total_negative_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_negative_map == (expected_map * (-1))), expected_map.n_elem);
}

TEST_F(SourceFindLabeling, 4CCL_ImageCross)
{
    real_t S = 10.0; // source pixel
    real_t n = 0.0; // empty pixel

    ccl_4connectivity = true;
    img = {
        { S, n, n, n, n, n, n, S },
        { n, S, n, n, n, n, S, n },
        { n, n, S, n, n, S, n, n },
        { n, n, n, S, n, n, n, n },
        { n, n, n, n, S, n, n, n },
        { n, n, S, n, n, S, n, n },
        { n, S, n, n, n, n, S, n },
        { S, n, n, n, n, n, n, S },
    };
    run();

#ifndef FFTSHIFT
    expected_map = {
        { 2, 0, 0, 0, 0, 0, 0, 14 },
        { 0, 4, 0, 0, 0, 0, 12, 0 },
        { 0, 0, 6, 0, 0, 10, 0, 0 },
        { 0, 0, 0, 7, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 8, 0, 0, 0 },
        { 0, 0, 5, 0, 0, 9, 0, 0 },
        { 0, 3, 0, 0, 0, 0, 11, 0 },
        { 1, 0, 0, 0, 0, 0, 0, 13 },
    };
#else
    expected_map = {
        { 1, 0, 0, 0, 0, 0, 0, 13 },
        { 0, 3, 0, 0, 0, 0, 11, 0 },
        { 0, 0, 5, 0, 0, 9, 0, 0 },
        { 0, 0, 0, 7, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 8, 0, 0, 0 },
        { 0, 0, 6, 0, 0, 10, 0, 0 },
        { 0, 4, 0, 0, 0, 0, 12, 0 },
        { 2, 0, 0, 0, 0, 0, 0, 14 },
    };
#endif
    expected_total_islands = 14;

    EXPECT_EQ(total_positive_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_positive_map == expected_map), expected_map.n_elem);

    EXPECT_EQ(total_negative_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_negative_map == (expected_map * (-1))), expected_map.n_elem);
}

TEST_F(SourceFindLabeling, 8CCL_ImageCross)
{
    real_t S = 10.0; // source pixel
    real_t n = 0.0; // empty pixel

    ccl_4connectivity = false;
    img = {
        { S, n, n, n, n, n, n, S },
        { n, S, n, n, n, n, S, n },
        { n, n, S, n, n, S, n, n },
        { n, n, n, S, n, n, n, n },
        { n, n, n, n, S, n, n, n },
        { n, n, S, n, n, S, n, n },
        { n, S, n, n, n, n, S, n },
        { S, n, n, n, n, n, n, S },
    };
    run();

#ifndef FFTSHIFT
    expected_map = {
        { 2, 0, 0, 0, 0, 0, 0, 3 },
        { 0, 2, 0, 0, 0, 0, 3, 0 },
        { 0, 0, 2, 0, 0, 3, 0, 0 },
        { 0, 0, 0, 2, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 2, 0, 0, 0 },
        { 0, 0, 1, 0, 0, 2, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 2, 0 },
        { 1, 0, 0, 0, 0, 0, 0, 2 },
    };
#else
    expected_map = {
        { 1, 0, 0, 0, 0, 0, 0, 3 },
        { 0, 1, 0, 0, 0, 0, 3, 0 },
        { 0, 0, 1, 0, 0, 3, 0, 0 },
        { 0, 0, 0, 1, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 1, 0, 0, 0 },
        { 0, 0, 2, 0, 0, 1, 0, 0 },
        { 0, 2, 0, 0, 0, 0, 1, 0 },
        { 2, 0, 0, 0, 0, 0, 0, 1 },
    };
#endif
    expected_total_islands = 3;

    EXPECT_EQ(total_positive_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_positive_map == expected_map), expected_map.n_elem);

    EXPECT_EQ(total_negative_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_negative_map == (expected_map * (-1))), expected_map.n_elem);
}

TEST_F(SourceFindLabeling, 4CCL_SplitObjs)
{
    real_t S = 10.0; // source pixel
    real_t n = 0.0; // empty pixel

    ccl_4connectivity = true;
    img = {
        { n, S, S, S, S, S, S, n },
        { n, n, n, S, S, n, n, n },
        { S, n, S, n, n, S, n, S },
        { S, n, n, n, n, n, n, S },
        { S, n, n, n, n, n, n, S },
        { S, n, n, n, n, n, n, S },
        { n, n, S, n, n, S, n, n },
        { n, S, S, S, S, S, S, n },
    };
    run();

#ifndef FFTSHIFT
    expected_map = {
        { 0, 3, 3, 3, 3, 3, 3, 0 },
        { 0, 0, 0, 3, 3, 0, 0, 0 },
        { 1, 0, 4, 0, 0, 5, 0, 6 },
        { 1, 0, 0, 0, 0, 0, 0, 6 },
        { 1, 0, 0, 0, 0, 0, 0, 6 },
        { 1, 0, 0, 0, 0, 0, 0, 6 },
        { 0, 0, 2, 0, 0, 2, 0, 0 },
        { 0, 2, 2, 2, 2, 2, 2, 0 },
    };
#else
    expected_map = {
        { 0, 2, 2, 2, 2, 2, 2, 0 },
        { 0, 0, 0, 2, 2, 0, 0, 0 },
        { 1, 0, 4, 0, 0, 5, 0, 6 },
        { 1, 0, 0, 0, 0, 0, 0, 6 },
        { 1, 0, 0, 0, 0, 0, 0, 6 },
        { 1, 0, 0, 0, 0, 0, 0, 6 },
        { 0, 0, 3, 0, 0, 3, 0, 0 },
        { 0, 3, 3, 3, 3, 3, 3, 0 },
    };
#endif
    expected_total_islands = 6;

    EXPECT_EQ(total_positive_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_positive_map == expected_map), expected_map.n_elem);

    EXPECT_EQ(total_negative_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_negative_map == (expected_map * (-1))), expected_map.n_elem);
}

TEST_F(SourceFindLabeling, 8CCL_SplitObjs)
{
    real_t S = 10.0; // source pixel
    real_t n = 0.0; // empty pixel

    ccl_4connectivity = false;
    img = {
        { n, S, S, S, S, S, S, n },
        { n, n, n, S, S, n, n, n },
        { S, n, S, n, n, S, n, S },
        { S, n, n, n, n, n, n, S },
        { S, n, n, n, n, n, n, S },
        { S, n, n, n, n, n, n, S },
        { n, n, S, n, n, S, n, n },
        { n, S, S, S, S, S, S, n },
    };
    run();

#ifndef FFTSHIFT
    expected_map = {
        { 0, 3, 3, 3, 3, 3, 3, 0 },
        { 0, 0, 0, 3, 3, 0, 0, 0 },
        { 1, 0, 3, 0, 0, 3, 0, 4 },
        { 1, 0, 0, 0, 0, 0, 0, 4 },
        { 1, 0, 0, 0, 0, 0, 0, 4 },
        { 1, 0, 0, 0, 0, 0, 0, 4 },
        { 0, 0, 2, 0, 0, 2, 0, 0 },
        { 0, 2, 2, 2, 2, 2, 2, 0 },
    };
#else
    expected_map = {
        { 0, 2, 2, 2, 2, 2, 2, 0 },
        { 0, 0, 0, 2, 2, 0, 0, 0 },
        { 1, 0, 2, 0, 0, 2, 0, 4 },
        { 1, 0, 0, 0, 0, 0, 0, 4 },
        { 1, 0, 0, 0, 0, 0, 0, 4 },
        { 1, 0, 0, 0, 0, 0, 0, 4 },
        { 0, 0, 3, 0, 0, 3, 0, 0 },
        { 0, 3, 3, 3, 3, 3, 3, 0 },
    };
#endif
    expected_total_islands = 4;

    EXPECT_EQ(total_positive_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_positive_map == expected_map), expected_map.n_elem);

    EXPECT_EQ(total_negative_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_negative_map == (expected_map * (-1))), expected_map.n_elem);
}

TEST_F(SourceFindLabeling, 4CCL_MixedObjs)
{
    real_t S = 10.0; // source pixel
    real_t I = -10.0; // source pixel
    real_t n = 0.0; // empty pixel

    ccl_4connectivity = true;
    img = {
        { n, n, n, n, n, n, n, n },
        { n, S, n, n, n, n, n, n },
        { n, n, S, S, n, n, n, n },
        { n, n, S, S, n, n, n, n },
        { n, n, n, n, S, n, n, I },
        { n, n, n, n, n, I, I, n },
        { n, n, n, n, n, I, I, n },
        { n, n, n, n, n, n, n, n },
    };
    run();

#ifndef FFTSHIFT
    expected_map = {
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 2, 2, 0, 0, 0, 0 },
        { 0, 0, 2, 2, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 3, 0, 0, -2 },
        { 0, 0, 0, 0, 0, -1, -1, 0 },
        { 0, 0, 0, 0, 0, -1, -1, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
    };
#else
    expected_map = {
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 2, 2, 0, 0, 0, 0 },
        { 0, 0, 2, 2, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 3, 0, 0, -2 },
        { 0, 0, 0, 0, 0, -1, -1, 0 },
        { 0, 0, 0, 0, 0, -1, -1, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
    };
#endif
    expected_total_islands = 5;

    EXPECT_EQ(total_positive_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_positive_map == expected_map), expected_map.n_elem);

    EXPECT_EQ(total_negative_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_negative_map == (expected_map * (-1))), expected_map.n_elem);
}

TEST_F(SourceFindLabeling, 8CCL_MixedObjs)
{
    real_t S = 10.0; // source pixel
    real_t I = -10.0; // source pixel
    real_t n = 0.0; // empty pixel

    ccl_4connectivity = false;
    img = {
        { n, n, n, n, n, n, n, n },
        { n, S, n, n, n, n, n, n },
        { n, n, S, S, n, n, n, n },
        { n, n, S, S, n, n, n, n },
        { n, n, n, n, S, n, n, I },
        { n, n, n, n, n, I, I, n },
        { n, n, n, n, n, I, I, n },
        { n, n, n, n, n, n, n, n },
    };
    run();

#ifndef FFTSHIFT
    expected_map = {
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 1, 1, 0, 0, 0, 0 },
        { 0, 0, 1, 1, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 1, 0, 0, -1 },
        { 0, 0, 0, 0, 0, -1, -1, 0 },
        { 0, 0, 0, 0, 0, -1, -1, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
    };
#else
    expected_map = {
        { 0, 0, 0, 0, 0, 0, 0, 0 },
        { 0, 1, 0, 0, 0, 0, 0, 0 },
        { 0, 0, 1, 1, 0, 0, 0, 0 },
        { 0, 0, 1, 1, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 1, 0, 0, -1 },
        { 0, 0, 0, 0, 0, -1, -1, 0 },
        { 0, 0, 0, 0, 0, -1, -1, 0 },
        { 0, 0, 0, 0, 0, 0, 0, 0 },
    };
#endif
    expected_total_islands = 2;

    EXPECT_EQ(total_positive_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_positive_map == expected_map), expected_map.n_elem);

    EXPECT_EQ(total_negative_islands, expected_total_islands);
    EXPECT_EQ(arma::accu(result_negative_map == (expected_map * (-1))), expected_map.n_elem);
}
