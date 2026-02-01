#pragma once

#include "core_types.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pc {

class PointCloud {
public:
  std::vector<position> positions;
  std::vector<color> colors;

  auto size() const { return positions.size(); }
  auto empty() const { return positions.empty(); }

  std::vector<std::byte> serialize(bool compress = false) const;
  static PointCloud deserialize(const std::vector<std::byte> &buffer);

private:
  std::vector<std::byte> compress() const;
  static PointCloud decompress(const std::vector<std::byte> &buffer,
                               unsigned long point_count);
};

PointCloud operator+(PointCloud const &lhs, PointCloud const &rhs);
PointCloud operator+=(PointCloud &lhs, const PointCloud &rhs);

struct PointCloudPacket {
  // out packet needs these explicitly sized types to ensure portability
  // between unix and windows systems
  uint64_t timestamp;
  uint64_t point_count;
  uint8_t compressed;
  std::vector<std::byte> data;
};

} // namespace pc