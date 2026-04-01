#pragma once

#include <string>

namespace pc {

struct FileConfiguration {
  std::string file_path = ""; // @file_opener
  bool load_sequence = false;
};

} // namespace pc
