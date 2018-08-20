#include "load_data.h"
#include <gtest/gtest.h>
#include <stp.h>

// Cmake variable
#ifndef _CNPYTESTPATH
#define _CNPYTESTPATH 0
#endif

using namespace stp;

// Cnpy test file locations
const std::string complex_1D("complex_1d.npy");
const std::string complex_2D("complex_2d.npy");
const std::string cnpy_data_path(_CNPYTESTPATH);

TEST(CnpyLoadComplex, complex1d)
{
    arma::cx_mat complex1d = load_npy_complex_array<double>(cnpy_data_path + complex_1D);

    // This is a row matrix
    arma::cx_mat expected_results
        = {
              (std::complex<double>(1.000000000000000000, 0.0000000000000000000)),
              (std::complex<double>(0.9497657153816386755, 0.3129617962077866355)),
              (std::complex<double>(0.8041098282287917343, 0.5944807685248221230)),
              (std::complex<double>(0.5776661771246112131, 0.8162731085894213701)),
              (std::complex<double>(0.2931852317082737636, 0.9560556573276295378)),
              (std::complex<double>(-0.02075161445913102642, 0.9997846620634562864)),
              (std::complex<double>(-0.3326035756124746112, 0.9430667322569473709)),
              (std::complex<double>(-0.6110393314010151844, 0.7916002371658312775)),
              (std::complex<double>(-0.8280848398163316304, 0.5606027988392140449)),
              (std::complex<double>(-0.9619338491686807435, 0.2732823994031187698))
          };

    // .st() transposes the row matrix (in order to compare with the column matrix complex1d)
    EXPECT_TRUE(arma::approx_equal(expected_results.st(), complex1d, "absdiff", 0.0));
}

TEST(CnpyLoadComplex, complex2d)
{
    arma::cx_mat complex2d = load_npy_complex_array<double>(cnpy_data_path + complex_2D);

    arma::cx_mat expected_results
        = {
              { (std::complex<double>(1.000000000000000000, 0.0000000000000000000)),
                  (std::complex<double>(0.9497657153816386755, 0.3129617962077866355)),
                  (std::complex<double>(0.8041098282287917343, 0.5944807685248221230)),
                  (std::complex<double>(0.5776661771246112131, 0.8162731085894213701)),
                  (std::complex<double>(0.2931852317082737636, 0.9560556573276295378)) },
              { (std::complex<double>(-0.02075161445913102642, 0.9997846620634562864)),
                  (std::complex<double>(-0.3326035756124746112, 0.9430667322569473709)),
                  (std::complex<double>(-0.6110393314010151844, 0.7916002371658312775)),
                  (std::complex<double>(-0.8280848398163316304, 0.5606027988392140449)),
                  (std::complex<double>(-0.9619338491686807435, 0.2732823994031187698)) }
          };

    EXPECT_TRUE(arma::approx_equal(expected_results, complex2d, "absdiff", 0.0));
}
