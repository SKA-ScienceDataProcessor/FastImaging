#include "load_json_config.h"

rapidjson::Document load_json_configuration(std::string& cfg)
{
    std::ifstream file(cfg);
    std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    rapidjson::Document aux;
    aux.Parse(str.c_str());

    return aux;
}
