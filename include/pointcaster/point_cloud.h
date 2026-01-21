#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pc {
struct float3 {
  float x = 0;
  float y = 0;
  float z = 0;

  constexpr float& operator[](std::size_t index) {
    return index == 0 ? x : (index == 1 ? y : z);
  }
  constexpr const float& operator[](std::size_t index) const {
    return index == 0 ? x : (index == 1 ? y : z);
  }
};

constexpr bool operator==(const float3 &lhs, const float3 &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}
constexpr bool operator!=(const float3 &lhs, const float3 &rhs) {
  return !(rhs == lhs);
}
}

namespace pc::types {

struct uint2 {
  unsigned int x, y = 0;

  bool operator==(const uint2 other) const {
    return x == other.x && y == other.y;
  }
  bool operator!=(const uint2 other) const { return !operator==(other); }
};

struct alignas(4) position {
  int16_t x = 0;
  int16_t y = 0;
  int16_t z = 0;
  int16_t __pad = 0;
};

constexpr bool operator==(const position &lhs, const position &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}
constexpr bool operator!=(const position &lhs, const position &rhs) {
  return !(rhs == lhs);
}

struct color {
  unsigned char r, g, b, a = 0;

  bool operator==(const color other) const {
    return r == other.r && g == other.g && b == other.b && a == other.a;
  }
  bool operator!=(const color other) const { return !operator==(other); }
};

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

} // namespace pc::types