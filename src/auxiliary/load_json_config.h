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

    int image_size = 0;
    double cell_size = 0.0;
    int kernel_support = 3;
    bool kernel_exact = false;
    int oversampling = 9;
    double detection_n_sigma = 0.0;
    double analysis_n_sigma = 0.0;

public:
    ConfigurationFile() = delete;

    ConfigurationFile(std::string cfg)
    {
        config_document = load_json_configuration(cfg);

        if (config_document.IsObject()) {
            if (config_document.HasMember("image_size_pix"))
                image_size = config_document["image_size_pix"].GetInt();
            if (config_document.HasMember("cell_size_arcsec"))
                cell_size = config_document["cell_size_arcsec"].GetDouble();
            if (config_document.HasMember("kernel_support"))
                kernel_support = config_document["kernel_support"].GetInt();
            if (config_document.HasMember("kernel_exact"))
                kernel_exact = config_document["kernel_exact"].GetBool();
            if (config_document.HasMember("oversampling"))
                oversampling = config_document["oversampling"].GetInt();
            if (config_document.HasMember("sourcefind_detection"))
                detection_n_sigma = config_document["sourcefind_detection"].GetDouble();
            if (config_document.HasMember("sourcefind_analysis"))
                analysis_n_sigma = config_document["sourcefind_analysis"].GetDouble();
        }
    }
};

#endif /* LOAD_JSON_CONFIG_H */
