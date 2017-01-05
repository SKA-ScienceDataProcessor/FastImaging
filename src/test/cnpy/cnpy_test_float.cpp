#include "../auxiliary/load_data.h"
#include <gtest/gtest.h>
#include <stp.h>

using namespace stp;

// Cnpy test file locations
const std::string FLOAT_1D_PATH("../data/cnpy_test/float_1d.npy");
const std::string FLOAT_2D_PATH("../data/cnpy_test/float_2d.npy");

TEST(CnpyLoadFloat, float1d)
{
    std::string location = __FILE__;
    location = location.substr(0, location.size() - strlen("cnpy_func_float.cpp"));

    cnpy::NpyArray float1d_npy = cnpy::npy_load(location.append(FLOAT_1D_PATH));
    arma::mat float1d = load_npy_double_array(float1d_npy);

    arma::mat expected_results
        = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

    EXPECT_TRUE(arma::approx_equal(expected_results.st(), float1d, "absdiff", 0.0));
}

TEST(CnpyLoadFloat, float2d)
{
    std::string location = __FILE__;
    location = location.substr(0, location.size() - strlen("cnpy_func_float.cpp"));

    cnpy::NpyArray float2d_npy = cnpy::npy_load(location.append(FLOAT_2D_PATH));
    arma::mat float2d = load_npy_double_array(float2d_npy);

    arma::mat expected_results
        = {
            { 0.0, 1.0, 2.0 },
            { 3.0, 4.0, 5.0 },
            { 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0 }
          };

    EXPECT_TRUE(arma::approx_equal(expected_results, float2d, "absdiff", 0.0));
}
