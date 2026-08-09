#pragma once

#include <pointcaster/core.h>
#include <string_view>
#include <unordered_map>

namespace pc {

// transparent hashing so lookups from a string_view don't allocate a key
struct PathHash {
  using is_transparent = void;
  [[nodiscard]] std::size_t operator()(std::string_view path) const noexcept {
    return std::hash<std::string_view>{}(path);
  }
};

template <typename T>
using StringMap = std::unordered_map<std::string, T, PathHash, std::equal_to<>>;

} // namespace pc