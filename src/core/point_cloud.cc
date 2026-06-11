#include <chrono>
#include <iostream>
#include <pointcaster/point_cloud.h>
#include <profiling/profiling_zone.h>
#include <zpp_bits.h>

namespace pc {

using namespace std::chrono;
using namespace pc::profiling;

auto PointCloud::serialize(bool compress) const -> std::vector<std::byte> {
  ProfilingZone zone("PointCloud::serialize");

  const auto timestamp = static_cast<uint64_t>(
      duration_cast<milliseconds>(system_clock::now().time_since_epoch())
          .count());
  const uint64_t point_count = size();
  const auto compression_flag = static_cast<uint8_t>(compress);
  std::vector<std::byte> buffer;

  if (compress) {
    std::vector<std::byte> payload;
    {
      ProfilingZone z("serialize::compress");
      payload = this->compress();
    }
    {
      ProfilingZone z("serialize::reserve");
      buffer.reserve(PointCloudPacket::header_bytes + payload.size());
    }
    {
      ProfilingZone z("serialize::write_payload");
      zpp::bits::out serializer{buffer};
      serializer(timestamp, point_count, compression_flag, payload).or_throw();
    }
  } else {
    {
      ProfilingZone z("serialize::reserve");
      buffer.reserve(PointCloudPacket::header_bytes +
                     positions.size() * sizeof(positions[0]) +
                     colors.size() * sizeof(colors[0]));
    }
    zpp::bits::out serializer{buffer};
    {
      ProfilingZone z("serialize::write_header");
      serializer(timestamp, point_count, compression_flag).or_throw();
    }
    {
      ProfilingZone z("serialize::write_body");
      serializer(*this).or_throw();
    }
  }
  return buffer;
}

auto PointCloud::deserialize(std::span<const std::byte> buffer) -> PointCloud {
  zpp::bits::in zpp_deserialize{buffer};

  uint64_t timestamp = 0;
  uint64_t point_count = 0;
  uint8_t compression_flag = 0;
  zpp_deserialize(timestamp, point_count, compression_flag).or_throw();

  if (compression_flag != 0) {
    std::vector<std::byte> payload;
    zpp_deserialize(payload).or_throw();
    return PointCloud::decompress(payload, point_count);
  }

  PointCloud point_cloud;
  zpp_deserialize(point_cloud).or_throw();
  return point_cloud;
}

PointCloud operator+(PointCloud const &lhs, PointCloud const &rhs) {
  std::vector<position> positions;
  std::vector<color> colors;

  const auto lhs_size = lhs.positions.size();
  const auto rhs_size = rhs.positions.size();

  positions.reserve(lhs_size + rhs_size);
  positions.insert(positions.end(), lhs.positions.begin(), lhs.positions.end());
  positions.insert(positions.end(), rhs.positions.begin(), rhs.positions.end());

  colors.reserve(lhs_size + rhs_size);
  colors.insert(colors.end(), lhs.colors.begin(), lhs.colors.end());
  colors.insert(colors.end(), rhs.colors.begin(), rhs.colors.end());

  return PointCloud{positions, colors};
}

PointCloud operator+=(PointCloud &lhs, PointCloud const &rhs) {
  const auto lhs_size = lhs.positions.size();
  const auto rhs_size = rhs.positions.size();

  lhs.positions.reserve(lhs_size + rhs_size);
  lhs.positions.insert(lhs.positions.end(), rhs.positions.begin(),
                       rhs.positions.end());

  lhs.colors.reserve(lhs_size + rhs_size);
  lhs.colors.insert(lhs.colors.end(), rhs.colors.begin(), rhs.colors.end());

  return lhs;
}

} // namespace pc