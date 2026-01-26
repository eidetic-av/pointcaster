#pragma once

#include <pointcaster/core.h>
#include <string>

namespace pc::uuid {

POINTCASTER_CORE_EXPORT const std::string word(int32_t min_chars = 4,
                                               int32_t max_chars = 9,
                                               std::string prefix = "",
                                               std::string suffix = "");

POINTCASTER_CORE_EXPORT long unsigned int digit();

} // namespace pc::uuid
