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

    rapidjson::Value rj_islands(rapidjson::kArrayType);

    for (auto&& i : sf.islands) {
        rapidjson::Value island(rapidjson::kObjectType);
        island.AddMember("sign", rapidjson::Value().SetInt(i.sign), allocator);
        island.AddMember("val", rapidjson::Value().SetDouble(i.extremum_val), allocator);
        island.AddMember("x_idx", rapidjson::Value().SetInt(i.extremum_x_idx), allocator);
        island.AddMember("y_idx", rapidjson::Value().SetInt(i.extremum_y_idx), allocator);
        island.AddMember("xbar", rapidjson::Value().SetDouble(i.xbar), allocator);
        island.AddMember("ybar", rapidjson::Value().SetDouble(i.ybar), allocator);
        rj_islands.PushBack(island, allocator);
    }
    doc.AddMember("Islands", rj_islands, allocator);

    // Write to file
    FILE* fp = fopen(filename.c_str(), "w");
    char writeBuffer[65536];
    rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
    rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
    doc.Accept(writer);
    fclose(fp);
}
