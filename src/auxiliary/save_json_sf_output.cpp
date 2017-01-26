#include "save_json_sf_output.h"
#include <cstdio>

// RapidJson
#include <rapidjson/document.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/writer.h>

void save_json_sourcefind_output(std::string& filename, stp::source_find_image& sf)
{
    rapidjson::Document doc;
    rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();
    doc.SetObject();

    rapidjson::Value island_vector(rapidjson::kArrayType);

    for (uint i = 0; i < sf.islands.size(); i++) {
        rapidjson::Value island(rapidjson::kObjectType);
        island.AddMember("sign", rapidjson::Value().SetInt(sf.islands[i].sign), allocator);
        island.AddMember("val", rapidjson::Value().SetDouble(sf.islands[i].extremum_val), allocator);
        island.AddMember("x_idx", rapidjson::Value().SetInt(sf.islands[i].extremum_x_idx), allocator);
        island.AddMember("y_idx", rapidjson::Value().SetInt(sf.islands[i].extremum_y_idx), allocator);
        island.AddMember("xbar", rapidjson::Value().SetDouble(sf.islands[i].xbar), allocator);
        island.AddMember("ybar", rapidjson::Value().SetDouble(sf.islands[i].ybar), allocator);
        island_vector.PushBack(island, allocator);
    }
    doc.AddMember("Islands", island_vector, allocator);

    // Write to file
    FILE* fp = fopen(filename.c_str(), "w");
    char writeBuffer[65536];
    rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
    rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
    doc.Accept(writer);
    fclose(fp);
}
