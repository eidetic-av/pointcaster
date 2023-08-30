#pragma once

#include <array>
#include <string>
#include <string_view>
#include <cctype>

namespace pc::strings {

constexpr std::string to_string(const std::string &str) noexcept { return str; }

constexpr std::string to_string(const char *str) noexcept {
  return str ? std::string(str) : std::string{};
}

constexpr std::string to_string(std::string_view str) noexcept {
  return str.empty() ? std::string{} : std::string(str);
}

constexpr std::string to_string(int value) noexcept {
  std::array<char, 32> buffer;
  auto len = std::snprintf(buffer.data(), buffer.size(), "%d", value);
  return std::string(buffer.data(), len);
}

constexpr std::string to_string(double value, int precision = 6) noexcept {
  std::array<char, 32> buffer;
  auto len =
      std::snprintf(buffer.data(), buffer.size(), "%.*f", precision, value);
  return std::string(buffer.data(), len);
}

template <typename... Args>
constexpr std::string concat(const Args &...args) noexcept {
  std::string result;
  result.reserve((to_string(args).size() + ...));
  ((result.append(to_string(args))), ...);
  return result;
}

std::string snake_case(const std::string &input) noexcept {
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

} // namespace pc::strings
