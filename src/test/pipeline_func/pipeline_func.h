/** @file pipeline_func.h
 *  @brief Header file for the pipeline tests
 *
 *  Reads the pipeline configuration json and
 *  loads the values onto variables
 *
 *  @bug No known bugs.
 */

#include <gtest/gtest.h>
#include <libstp.h>

// RapidJson
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

#include <load_data.h>

// Cmake variable
#ifndef _TESTPATH
#define _TESTPATH 0
#endif

rapidjson::Value set_up_json(const char* typeConvolution, const char* typeTest)
{
    // Loads the json file
    std::ifstream file("conf_tests.json");
    std::string str((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    // Parses the json to the node with the our convolution and test type
    rapidjson::Document aux;
    aux.Parse(str.c_str());

    std::string path_file;
    if (_TESTPATH) {
        path_file = _TESTPATH;
    } else {
        path_file = aux["path"].GetString();
    }
    rapidjson::Value d;
    d = aux[typeConvolution][typeTest];

    std::string path_expected_str = (path_file + d["expected_results"].GetString());
    std::string path_input_str = (path_file + d["input_file"].GetString());

    d["expected_results"].SetString(path_expected_str.c_str(), aux.GetAllocator());
    d["input_file"].SetString(path_input_str.c_str(), aux.GetAllocator());

    return d;
}

struct PipelineHandler {
    rapidjson::Value d;
    arma::cx_cube result;
    arma::cx_cube expected_result;

    double image_size;
    double cell_size;
    double oversampling;
    int support;
    bool pad;
    bool normalize;

public:
    PipelineHandler(const char* typeConvolution, const char* typeTest)
    {
        d = set_up_json(typeConvolution, typeTest);
        image_size = d["image_size"].GetDouble();
        cell_size = d["cell_size"].GetDouble();
        support = d["support"].GetInt();
        oversampling = d["oversampling"].GetDouble();
        pad = d["pad"].GetBool();
        normalize = d["normalize"].GetBool();
    }
};
