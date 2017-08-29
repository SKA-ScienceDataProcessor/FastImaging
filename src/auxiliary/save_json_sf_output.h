#ifndef SAVE_JSON_SF_OUTPUT_H
#define SAVE_JSON_SF_OUTPUT_H

#include <stp.h>
#include <string>

/**
 * @brief Saves the sourcefind island parameters in the given json file
 *
 * @param[in] filename (string): Output filename where json data will be written.
 * @param[in] sf (stp::SourceFindImage): Source find image structure containing list of islands to be saved.
 *
 */
void save_json_sourcefind_output(std::string& filename, stp::SourceFindImage& sf);

#endif /* SAVE_JSON_SF_OUTPUT_H */
