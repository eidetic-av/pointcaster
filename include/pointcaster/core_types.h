#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace pc {

using int2 = std::pair<int, int>;

struct float3 {
  float x = 0;
  float y = 0;
  float z = 0;
  auto operator<=>(const float3 &f) const = default;
};

struct float4 {
  float x = 0;
  float y = 0;
  float z = 0;
  float w = 0;
  auto operator<=>(const float4 &f) const = default;
};

struct float4x4 {
  std::array<float, 16> values{
      1, 0, 0, 0, //
      0, 1, 0, 0, //
      0, 0, 1, 0, //
      0, 0, 0, 1,
  };
  bool operator==(const float4x4 &) const = default;
};

struct quaternion {
  float scalar = 1;
  float x = 0;
  float y = 0;
  float z = 0;
  auto operator<=>(const quaternion &q) const = default;
};

struct uint2 {
  unsigned int x = 0, y = 0;
  auto operator<=>(const uint2 &u) const = default;
};

struct alignas(4) short3 {
  int16_t x = 0;
  int16_t y = 0;
  int16_t z = 0;
  int16_t __pad = 0;
  auto operator<=>(const short3 &s) const = default;
};

struct alignas(4) position {
  int16_t x = 0;
  int16_t y = 0;
  int16_t z = 0;
  int16_t __pad = 0;
  auto operator<=>(const position &p) const = default;
};

struct position_bounds {
  position min{std::numeric_limits<int16_t>::max(),
               std::numeric_limits<int16_t>::max(),
               std::numeric_limits<int16_t>::max()};
  position max{std::numeric_limits<int16_t>::min(),
               std::numeric_limits<int16_t>::min(),
               std::numeric_limits<int16_t>::min()};
};

struct color {
  unsigned char r = 0, g = 0, b = 0, a = 0;
  auto operator<=>(const color &c) const = default;
};

struct color_rgb {
  unsigned char r = 0;
  unsigned char g = 0;
  unsigned char b = 0;
  auto operator<=>(const color_rgb &c) const = default;

  operator color() const { return {r, g, b, 255}; };
};

} // namespace pc