#pragma once

#include "wtf/wtf.h"
#include <string>

namespace pc::uuid {

const std::string word(int32_t min_chars = 4, int32_t max_chars = 9,
                       std::string prefix = "", std::string suffix = "");

} // namespace pc::uuid
