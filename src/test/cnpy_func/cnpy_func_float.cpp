#include <gtest/gtest.h>
#include <libstp.h>
#include <load_data.h>

// Cnpy test file locations
const char* FLOAT_1D_PATH("../data/cnpy_test/float_1d.npy");
const char* FLOAT_2D_PATH("../data/cnpy_test/float_2d.npy");

TEST(CnpyLoadFloat, float1d)
{
    std::string location = __FILE__;
    location = location.substr(0, location.size() - strlen("cnpy_func_float.cpp"));

    cnpy::NpyArray float1d_array = cnpy::npy_load(location.append(FLOAT_1D_PATH));
    arma::cx_mat float1d = load_npy_array(&float1d_array);

    arma::cx_mat expected_results
        = {
            { std::complex<double>(0.0, 0.0) },
            { std::complex<double>(1.0, 0.0) },
            { std::complex<double>(2.0, 0.0) },
            { std::complex<double>(3.0, 0.0) },
            { std::complex<double>(4.0, 0.0) },
            { std::complex<double>(5.0, 0.0) },
            { std::complex<double>(6.0, 0.0) },
            { std::complex<double>(7.0, 0.0) },
            { std::complex<double>(8.0, 0.0) },
            { std::complex<double>(9.0, 0.0) }
          };

    EXPECT_TRUE(arma::approx_equal(expected_results, float1d, "absdiff", 0.0));
}

TEST(CnpyLoadFloat, float2d)
{
    std::string location = __FILE__;
    location = location.substr(0, location.size() - strlen("cnpy_func_float.cpp"));

    cnpy::NpyArray float2d_array = cnpy::npy_load(location.append(FLOAT_2D_PATH));
    arma::cx_mat float2d = load_npy_array(&float2d_array);

    arma::cx_mat expected_results
        = {
            { std::complex<double>(0.0, 0.0), std::complex<double>(1.0, 0.0), std::complex<double>(2.0, 0.0) },
            { std::complex<double>(3.0, 0.0), std::complex<double>(4.0, 0.0), std::complex<double>(5.0, 0.0) },
            { std::complex<double>(6.0, 0.0), std::complex<double>(7.0, 0.0), std::complex<double>(8.0, 0.0) },
            { std::complex<double>(9.0, 0.0), std::complex<double>(10.0, 0.0), std::complex<double>(11.0, 0.0) }
          };

    EXPECT_TRUE(arma::approx_equal(expected_results, float2d, "absdiff", 0.0));
}
