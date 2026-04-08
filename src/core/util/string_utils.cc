#include "string_utils.h"

#include <array>
#include <charconv>
#include <expected>
#include <ranges>
#include <string_view>
#include <system_error>

namespace pc::util {

std::expected<std::array<uint8_t, 4>, std::string>
parse_ip_string(std::string_view sv) {
  std::array<uint8_t, 4> output{};
  std::size_t node_index = 0;

  for (auto node : std::views::split(sv, '.')) {

    if (node_index >= 4) return std::unexpected("too many address nodes");
    if (node.begin() == node.end())
      return std::unexpected("empty address node");

    int node_value = 0;
    auto node_end = node.data() + node.size();
    auto [ptr, ec] = std::from_chars(node.data(), node_end, node_value);

    if (ec != std::errc{} || ptr != node_end)
      return std::unexpected(
          std::format("address node {} is not a number", node_index));

    if (node_value < 0 || node_value > 255)
      return std::unexpected(
          std::format("address node {} is out of range", node_index));

    output[node_index++] = static_cast<uint8_t>(node_value);
  }
  if (node_index != 4) return std::unexpected("wrong number of address nodes");

  return output;
}

} // namespace pc::util