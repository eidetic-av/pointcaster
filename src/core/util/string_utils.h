#pragma once

#include <expected>
#include <pointcaster/core.h>
#include <string_view>

namespace pc::util {

POINTCASTER_CORE_EXPORT std::expected<std::array<uint8_t, 4>, std::string>
parse_ip_string(std::string_view sv);

} // namespace pc::util