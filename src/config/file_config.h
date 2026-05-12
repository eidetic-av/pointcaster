#pragma once

#include <string>

namespace pc {

struct FileConfiguration {
  std::string path = ""; // @file_opener
};

struct FileFolderConfiguration {
  std::string path = ""; // @file_opener @folder_opener
};

} // namespace pc
