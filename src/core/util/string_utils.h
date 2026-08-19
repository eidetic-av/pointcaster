#pragma once

#include <concepts>
#include <expected>
#include <format>
#include <pointcaster/core.h>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_set>

namespace pc::util {

POINTCASTER_CORE_EXPORT std::expected<std::array<uint8_t, 4>, std::string>
parse_ip_string(std::string_view sv);

constexpr std::string to_snake_case(const std::string_view input) {
  constexpr auto is_upper = [](char c) { return c >= 'A' && c <= 'Z'; };
  constexpr auto is_lower = [](char c) { return c >= 'a' && c <= 'z'; };
  constexpr auto is_digit = [](char c) { return c >= '0' && c <= '9'; };

  std::string result;
  result.reserve(input.size() + input.size() / 4);
  for (size_t i = 0; i < input.size(); ++i) {
    const char c = input[i];
    if (i > 0 && is_upper(c) &&
        (is_lower(input[i - 1]) || is_digit(input[i - 1])))
      result.push_back('_');
    result.push_back(is_upper(c) ? static_cast<char>(c - 'A' + 'a') : c);
  }
  return result;
}

// the lowest numbered "<prefix>_<n>" (counting from 1) that doesn't already
// appear in existing_labels
template <std::ranges::input_range Labels>
  requires std::convertible_to<std::ranges::range_reference_t<Labels>,
                               std::string_view>
std::string next_available_label(std::string_view prefix,
                                 Labels &&existing_labels) {
  std::unordered_set<std::string> taken;
  for (auto &&label : existing_labels) taken.emplace(label);
  for (std::size_t index = 1;; ++index) {
    auto candidate = std::format("{}_{}", prefix, index);
    if (!taken.contains(candidate)) return candidate;
  }
}

} // namespace pc::util
