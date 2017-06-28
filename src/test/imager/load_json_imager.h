/** @file imager_func.h
 *  @brief Header file for the imager tests
 *
 *  Reads the imager configuration json and
 *  loads the values onto variables
 *
 *  @bug No known bugs.
 */

#include <stp.h>

#include "../../auxiliary/load_data.h"
// RapidJson
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

// Cmake variable
#ifndef _IMAGER_TESTPATH
#define _IMAGER_TESTPATH 0
#endif

rapidjson::Value set_up_json(const std::string& typeConvolution, const std::string& typeTest);

struct ImagerHandler {
    rapidjson::Value val;
    std::pair<arma::cx_mat, arma::cx_mat> expected_result;
    std::pair<arma::mat, arma::mat> result;

    double image_size;
    double cell_size;
    bool kernel_exact;
    int oversampling;
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
        , kernel_exact(val["kernel_exact"].GetBool())
        , oversampling(val["oversampling"].GetInt())
        , pad(val["pad"].GetBool())
        , normalize(val["normalize"].GetBool())
    {
    }
};
