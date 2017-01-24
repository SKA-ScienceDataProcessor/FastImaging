#ifndef LOAD_JSON_CONFIG_H
#define LOAD_JSON_CONFIG_H

#include <fstream>

// RapidJson
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

/**
 * @brief Loads a json configuration file into a rapidjson document
 *
 * Uses ifstream to load the json file, and after this parses the result into a document.
 *
 * @param[in] cfg (string): Input config json filename.
 *
 * @return A rapidjson::Document with the content of a json file.
 */
rapidjson::Document
load_json_configuration(std::string& cfg);

/**
 * @brief The Configuration file struct
 *
 * Load all parameters in json configuration file to an object
 */
struct ConfigurationFile {
    rapidjson::Document config_document;

    int image_size;
    double cell_size;
    int kernel_support;
    bool kernel_exact;
    int oversampling;
    double detection_n_sigma;
    double analysis_n_sigma;

public:
    ConfigurationFile() = delete;

    ConfigurationFile(std::string cfg)
    {
        config_document = load_json_configuration(cfg);

        image_size = config_document["image_size_pix"].GetInt();
        cell_size = config_document["cell_size_arcsec"].GetDouble();
        kernel_support = config_document["kernel_support"].GetInt();
        kernel_exact = config_document["kernel_exact"].GetBool();
        oversampling = config_document["oversampling"].GetInt();
        detection_n_sigma = config_document["sourcefind_detection"].GetDouble();
        analysis_n_sigma = config_document["sourcefind_analysis"].GetDouble();
    }
};

#endif /* LOAD_JSON_CONFIG_H */
