/** @file imager_func.h
 *  @brief Header file for the imager tests
 *
 *  Reads the imager configuration json and
 *  loads the values onto variables
 *
 *  @bug No known bugs.
 */

#include "load_data.h"
// RapidJson
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

// Cmake variable
#ifndef _TESTPATH
#define _TESTPATH 0
#endif

rapidjson::Value set_up_json(const std::string& typeConvolution, const std::string& typeTest);

struct ImagerHandler {
    rapidjson::Value val;
    std::pair<arma::cx_mat, arma::cx_mat> expected_result;
    std::pair<arma::cx_mat, arma::cx_mat> result;

    double image_size;
    double cell_size;
    bool use_oversampling;
    int oversampling_val;
    int support;
    bool pad;
    bool normalize;

public:
    ImagerHandler() = default;
    ImagerHandler(const std::string& typeConvolution, const std::string& typeTest)
        : val(set_up_json(typeConvolution, typeTest))
        , image_size(val["image_size"].GetDouble())
        , cell_size(val["cell_size"].GetDouble())
        , support(val["support"].GetInt())
        , use_oversampling(val["use_oversampling"].GetBool())
        , oversampling_val(val["oversampling_val"].GetInt())
        , pad(val["pad"].GetBool())
        , normalize(val["normalize"].GetBool())
    {
    }
};
