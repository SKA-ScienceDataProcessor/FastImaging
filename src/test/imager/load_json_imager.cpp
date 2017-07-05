/** @file imager_func.cpp
 *  @brief cpp file for the imager tests
 *
 *  Reads the imager configuration json and
 *  loads the values onto variables
 *
 *  @bug No known bugs.
 */

#include "load_json_imager.h"

rapidjson::Value ImagerHandler::set_up_json(const std::string& typeConvolution, const std::string& typeTest)
{
    // Loads the json file
    std::ifstream file("conf_tests.json");
    std::string str((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    file.close();

    // Parses the json file
    rapidjson::Document doc_aux;
    doc_aux.Parse(str.c_str());

    std::string path_file;
    if (_IMAGER_TESTPATH) {
        path_file = _IMAGER_TESTPATH;
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
