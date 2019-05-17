/** @file load_json_imager.h
 *  @brief Header file for the imager tests
 *
 *  Reads the imager configuration json and
 *  loads the values onto variables
 */

#include <stp.h>

#include <load_data.h>
// RapidJson
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

// Cmake variable
#ifndef _IMAGER_TESTPATH
#define _IMAGER_TESTPATH 0
#endif

class ImagerHandler {
public:
    rapidjson::Value val;
    std::pair<arma::mat, arma::mat> expected_result;
    std::pair<arma::mat, arma::mat> result;

    double image_size;
    double cell_size;
    double padding_factor;
    bool kernel_exact;
    int oversampling;
    int support;
    bool gen_beam;
    bool gridding_correction;
    bool analytic_gcf;

    ImagerHandler() = default;
    ImagerHandler(const std::string& typeConvolution, const std::string& typeTest)
        : val(set_up_json(typeConvolution, typeTest))
        , image_size(val["image_size"].GetDouble())
        , cell_size(val["cell_size"].GetDouble())
        , padding_factor(1.0)
        , support(val["support"].GetInt())
        , kernel_exact(val["kernel_exact"].GetBool())
        , oversampling(val["oversampling"].GetInt())
        , gen_beam(true)
        , gridding_correction(val["gridding_correction"].GetBool())
        , analytic_gcf(val["analytic_gcf"].GetBool())
    {
    }

private:
    rapidjson::Value set_up_json(const std::string& typeConvolution, const std::string& typeTest);
};
