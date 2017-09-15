#include "save_json_sf_output.h"
#include <cstdio>

// RapidJson
#include <rapidjson/document.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/writer.h>

void save_json_sourcefind_output(std::string& filename, stp::SourceFindImage& sf)
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
        rapidjson::Value moments_fit(rapidjson::kObjectType);
        moments_fit.AddMember("amplitude", rapidjson::Value().SetDouble(i.moments_fit.amplitude), allocator);
        moments_fit.AddMember("x_centre", rapidjson::Value().SetDouble(i.moments_fit.x_centre), allocator);
        moments_fit.AddMember("y_centre", rapidjson::Value().SetDouble(i.moments_fit.y_centre), allocator);
        moments_fit.AddMember("semimajor", rapidjson::Value().SetDouble(i.moments_fit.semimajor), allocator);
        moments_fit.AddMember("semiminor", rapidjson::Value().SetDouble(i.moments_fit.semiminor), allocator);
        moments_fit.AddMember("theta", rapidjson::Value().SetDouble(i.moments_fit.theta), allocator);
        island.AddMember("MomentsFit", moments_fit, allocator);
        if (sf.fit_gaussian) {
            rapidjson::Value leastsq_fit(rapidjson::kObjectType);
            leastsq_fit.AddMember("amplitude", rapidjson::Value().SetDouble(i.leastsq_fit.amplitude), allocator);
            leastsq_fit.AddMember("x_centre", rapidjson::Value().SetDouble(i.leastsq_fit.x_centre), allocator);
            leastsq_fit.AddMember("y_centre", rapidjson::Value().SetDouble(i.leastsq_fit.y_centre), allocator);
            leastsq_fit.AddMember("semimajor", rapidjson::Value().SetDouble(i.leastsq_fit.semimajor), allocator);
            leastsq_fit.AddMember("semiminor", rapidjson::Value().SetDouble(i.leastsq_fit.semiminor), allocator);
            leastsq_fit.AddMember("theta", rapidjson::Value().SetDouble(i.leastsq_fit.theta), allocator);
            island.AddMember("GaussianFit", leastsq_fit, allocator);
        }
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
