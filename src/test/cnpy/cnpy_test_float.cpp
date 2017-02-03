#include "load_data.h"
#include <gtest/gtest.h>
#include <stp.h>

// Cmake variable
#ifndef _CNPYTESTPATH
#define _CNPYTESTPATH 0
#endif

using namespace stp;

// Cnpy test file locations
const std::string float_1D("float_1d.npy");
const std::string float_2D("float_2d.npy");
const std::string cnpy_data_path(_CNPYTESTPATH);

TEST(CnpyLoadFloat, float1d)
{
    arma::mat float1d = load_npy_double_array(cnpy_data_path + float_1D);

    arma::mat expected_results
        = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

    EXPECT_TRUE(arma::approx_equal(expected_results.st(), float1d, "absdiff", 0.0));
}

TEST(CnpyLoadFloat, float2d)
{
    arma::mat float2d = load_npy_double_array(cnpy_data_path + float_2D);

    arma::mat expected_results
        = {
            { 0.0, 1.0, 2.0 },
            { 3.0, 4.0, 5.0 },
            { 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0 }
          };

    EXPECT_TRUE(arma::approx_equal(expected_results, float2d, "absdiff", 0.0));
}
