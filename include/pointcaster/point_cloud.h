#pragma once

#include "core_types.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <pointcaster/core.h>
#include <span>
#include <vector>

namespace pc {

class PointCloud {
public:
  std::vector<position> positions;
  std::vector<color> colors;
  position_bounds bounds;

  auto size() const { return positions.size(); }
  auto empty() const { return positions.empty(); }

  void resize(std::size_t new_size) {
    positions.resize(new_size);
    colors.resize(new_size);
  }

  void reserve(std::size_t new_capacity) {
    positions.reserve(new_capacity);
    colors.reserve(new_capacity);
  }

  POINTCASTER_CORE_EXPORT std::vector<std::byte>
  serialize(bool compress = false) const;

  POINTCASTER_CORE_EXPORT static PointCloud
  deserialize(std::span<const std::byte> buffer);

private:
  std::vector<std::byte> compress() const;
  static PointCloud decompress(const std::vector<std::byte> &buffer,
                               unsigned long point_count);
};

POINTCASTER_CORE_EXPORT PointCloud operator+(PointCloud const &lhs,
                                              PointCloud const &rhs);
POINTCASTER_CORE_EXPORT PointCloud operator+=(PointCloud &lhs,
                                               const PointCloud &rhs);

using PointCloudRef = std::reference_wrapper<PointCloud>;

constexpr bool operator==(const PointCloudRef &lhs, const PointCloudRef &rhs) {
  return std::addressof(lhs.get()) == std::addressof(rhs.get());
}
constexpr bool operator!=(const PointCloudRef &lhs, const PointCloudRef &rhs) {
  return !(rhs == lhs);
}

struct PointCloudPacket {
  // out packet needs these explicitly sized types to ensure portability
  // between unix and windows systems
  uint64_t timestamp;
  uint64_t point_count;
  uint8_t compressed;
  std::vector<std::byte> data;

  static constexpr std::size_t header_bytes =
      sizeof(uint64_t)   // timestamp
      + sizeof(uint64_t) // point_count
      + sizeof(uint8_t); // compressed flag
};

} // namespace pc