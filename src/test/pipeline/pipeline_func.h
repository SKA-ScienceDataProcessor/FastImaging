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
#include <benchmark/benchmark.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

#include <load_data.h>

// Cmake variable
#ifndef _TESTPATH
#define _TESTPATH 0
#endif

rapidjson::Value set_up_json(const std::string typeConvolution, const std::string typeTest)
{
    // Loads the json file
    std::ifstream file("conf_tests.json");
    std::string str((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    file.close();

    // Parses the json to the node with the our convolution and test type
    rapidjson::Document doc_aux;
    doc_aux.Parse(str.c_str());

    std::string path_file;
    if (_TESTPATH) {
        path_file = _TESTPATH;
    } else {
        path_file = doc_aux["path"].GetString();
    }
    rapidjson::Value val;
    val = doc_aux[typeConvolution.c_str()][typeTest.c_str()];

    std::string path_expected_str = (path_file + val["expected_results"].GetString());
    std::string path_input_str = (path_file + val["input_file"].GetString());

    val["expected_results"].SetString(path_expected_str.c_str(), doc_aux.GetAllocator());
    val["input_file"].SetString(path_input_str.c_str(), doc_aux.GetAllocator());

    return val;
}

struct PipelineHandler {
    rapidjson::Value val;
    arma::cx_cube result;
    arma::cx_cube expected_result;

    double image_size;
    double cell_size;
    double oversampling;
    int support;
    bool pad;
    bool normalize;

public:
    PipelineHandler(const std::string typeConvolution, const std::string typeTest)
    {
        val = set_up_json(typeConvolution, typeTest);
        image_size = val["image_size"].GetDouble();
        cell_size = val["cell_size"].GetDouble();
        support = val["support"].GetInt();
        oversampling = val["oversampling"].GetDouble();
        pad = val["pad"].GetBool();
        normalize = val["normalize"].GetBool();
    }
};
