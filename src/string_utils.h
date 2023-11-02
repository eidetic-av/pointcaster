#pragma once

#include <array>
#include <string>
#include <string_view>
#include <cctype>
#include <unordered_map>

namespace pc::strings {

inline std::string to_string(const std::string &str) noexcept { return str; }

inline std::string to_string(const char *str) noexcept {
  return str ? std::string(str) : std::string{};
}

inline std::string to_string(std::string_view str) noexcept {
  return str.empty() ? std::string{} : std::string(str);
}

inline std::string to_string(int value) noexcept {
  std::array<char, 32> buffer;
  auto len = std::snprintf(buffer.data(), buffer.size(), "%d", value);
  return std::string(buffer.data(), len);
}

inline std::string to_string(double value, int precision = 6) noexcept {
  std::array<char, 32> buffer;
  auto len =
      std::snprintf(buffer.data(), buffer.size(), "%.*f", precision, value);
  return std::string(buffer.data(), len);
}

template <typename... Args>
std::string concat(const Args &...args) noexcept {
  std::string result;
  result.reserve((to_string(args).size() + ...));
  ((result.append(to_string(args))), ...);
  return result;
}

inline std::string snake_case(std::string_view input) noexcept {
  std::string result;
  result.reserve(input.size());
  for (auto c : input) {
    if (std::isalnum(static_cast<unsigned char>(c))) {
      result.push_back(std::tolower(static_cast<unsigned char>(c)));
    } else if (std::isspace(static_cast<unsigned char>(c))) {
      result.push_back('_');
    }
  }
  return result;
}

inline std::string title_case(std::string_view input) noexcept {
  std::string result;
  result.reserve(input.size());
  bool capitalize = true;
  for (auto c : input) {
    if (c == '_') {
      result.push_back(' '); // replace underscore with space
      capitalize = true;
    } else if (capitalize) {
      result.push_back(std::toupper(static_cast<unsigned char>(c)));
      capitalize = false;
    } else {
      result.push_back(c);
    }
  }
  return result;
}

inline std::string sentence_case(std::string_view input) noexcept {
  std::string result;
  result.reserve(input.size());
  bool capitalize = true;
  for (auto c : input) {
    if (c == '_') {
      result.push_back(' '); // replace underscore with space
    } else if (capitalize) {
      result.push_back(std::toupper(static_cast<unsigned char>(c)));
      capitalize = false;
    } else {
      result.push_back(c);
    }
  }
  return result;
}

inline constexpr std::string_view last_element(std::string_view str,
				     char delimiter = '.') noexcept {
  if (auto pos = str.rfind(delimiter); pos != std::string_view::npos) {
    return str.substr(pos + 1);
  }
  return str;
}

} // namespace pc::strings
