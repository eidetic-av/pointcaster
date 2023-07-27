#include "path.h"
#include <fstream>
#include <iterator>
#include <string>
#ifdef _WIN32
#include <Windows.h>
#include <wchar.h>
#endif

namespace pc::path {

std::filesystem::path exe_path() {
#ifdef _WIN32
  wchar_t buffer[MAX_PATH];
  GetModuleFileNameW(NULL, buffer, sizeof(buffer));
  return std::filesystem::path(buffer);
#else
  return std::filesystem::read_symlink("/proc/self/exe");
#endif
}

std::filesystem::path exe_directory() { return exe_path().parent_path(); }

std::filesystem::path data_directory() { return exe_directory() / "data"; }

std::filesystem::path get_or_create_data_directory() {
  auto dir = data_directory();
  if (!std::filesystem::exists(dir))
    std::filesystem::create_directory(dir);
  return dir;
}

std::vector<uint8_t> load_file(std::filesystem::path file_path) {
  std::ifstream file(file_path, std::ios::binary);
  
  std::streampos file_size;
  file.seekg(0, std::ios::end);
  file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer;
  buffer.reserve(file_size);
  buffer.assign(std::istreambuf_iterator<char>(file),
		std::istreambuf_iterator<char>());

  return buffer;
}

bool save_file(std::filesystem::path file_path, std::vector<uint8_t> buffer) {
  std::ofstream output_file(file_path.c_str(), std::ios::out | std::ios::binary);
  if (!output_file.is_open()) return false;
  output_file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
  return true;
}

} // namespace path
