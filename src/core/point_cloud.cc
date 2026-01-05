#include <chrono>
#include <iostream>
#include <pointcaster/point_cloud.h>
#include <zpp_bits.h>

namespace pc::types {

using namespace std::chrono;

auto PointCloud::serialize(bool compress) const -> std::vector<std::byte> {
  auto now = system_clock::now().time_since_epoch();
  uint64_t point_count = size();

  if (compress) {
    // PointCloud::compress is implemented by whatever codec we build the lib
    // with
    PointCloudPacket packet{
        static_cast<uint64_t>(duration_cast<milliseconds>(now).count()),
        point_count, static_cast<uint8_t>(compress), this->compress()};
    auto [output_data, zpp_serialize] = zpp::bits::data_out();
    zpp_serialize(packet).or_throw();
    return output_data;
  }

  auto [point_cloud_bytes, serialize_inner] = zpp::bits::data_out();
  serialize_inner(*this).or_throw();

  PointCloudPacket packet{
      static_cast<uint64_t>(duration_cast<milliseconds>(now).count()),
      point_count, static_cast<uint8_t>(compress),
      std::move(point_cloud_bytes)};
  auto [output_data, zpp_serialize] = zpp::bits::data_out();
  zpp_serialize(packet).or_throw();

  return output_data;
}

auto PointCloud::deserialize(const std::vector<std::byte> &buffer)
    -> PointCloud {
  // first, deserialize to a PointCloudPacket
  PointCloudPacket packet;
  auto zpp_deserialize = zpp::bits::in(buffer);
  zpp_deserialize(packet).or_throw();

  // then decode the internal data buffer differently based on whether the
  // packet is compressed or not, and move the resulting points and colors
  // into a PointCloud class instance
  if (packet.compressed) {
    // PointCloud::decompress is implemented by whatever codec we choose to
    // build the lib with
    auto point_count = packet.point_count;
    return PointCloud::decompress(packet.data, point_count);
  }
  if (packet.compressed)
    std::cerr << "You are trying to deserialize a compressed \
	PointCloudPacket, but this library was built without codec support";

  // if it's not compressed, our packet contains a serialized PointCloud
  PointCloud point_cloud;
  auto deserialize_inner = zpp::bits::in(packet.data);
  deserialize_inner(point_cloud).or_throw();
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

} // namespace pc::types