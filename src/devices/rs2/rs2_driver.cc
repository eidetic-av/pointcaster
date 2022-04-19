#include "rs2_driver.h"
#include <cstdint>
#include <librealsense2/hpp/rs_processing.hpp>
#include <limits>
#include <spdlog/spdlog.h>

namespace bob::sensors {

using namespace Magnum;
using namespace std::chrono_literals;

Rs2Driver::Rs2Driver(int device_index_) {
  device_index = device_index_;
  open();
}

Rs2Driver::~Rs2Driver() { close(); }

bool Rs2Driver::open() {
  auto device_list = _context.query_devices(RS2_PRODUCT_LINE_ANY);
  if (device_list.size() < 1) return false;
  _device = device_list.front();
  spdlog::info("Found {} Rs2 devices", device_list.size());
  spdlog::info("  -> Enabling stream for device at index 0");
  _config.enable_stream(RS2_STREAM_COLOR);
  _config.enable_stream(RS2_STREAM_DEPTH);
  // _config.enable_all_streams();
  _pipe.start(_config);

  // start a thread that captures new frames and dumps them into raw buffers
  std::thread capture_loop([&]() {
    rs2::pointcloud point_cloud;
    rs2::points points;
    while (isOpen()) {
      // just block until we receive both depth and color data
      auto frames = _pipe.wait_for_frames();
      const rs2::depth_frame depth_frame = frames.get_depth_frame();
      // generate the pointcloud and texture mappings
      points = point_cloud.calculate(depth_frame);
      const rs2::video_frame color_frame = frames.get_color_frame();

      // tell point cloud to map to this color frame
      point_cloud.map_to(color_frame);

      auto vertices = points.get_vertices();

      // move points into buffers
      std::lock_guard<std::mutex> lock(_buffer_mutex);
      _point_count = depth_frame.get_width() * depth_frame.get_height();
      const size_t buffer_float_count = _point_count * 3;
      _positions_buffer.resize(buffer_float_count);
      memcpy(_positions_buffer.data(), vertices, buffer_float_count * sizeof(float));

      // spdlog::info(point_buffer.get_vertices()[1000].x);

      _colors_buffer.resize(_point_count);

      _buffers_updated = true;
    }
  });
  capture_loop.detach();
  _open = true;
  return true;
}

bool Rs2Driver::close() {
  _open = false;
  // _device.close();
  return true;
}

PointCloud Rs2Driver::getPointCloud(const DeviceConfiguration& config) {
  if (!_buffers_updated)
    return _point_cloud;
  std::lock_guard<std::mutex> lock(_buffer_mutex);

  std::vector<position> positions(_point_count);
  std::vector<float> colors(_point_count);
  // memcpy(colors.data(), _colors_buffer.data(), _point_count * sizeof(float));
  auto colors_input = reinterpret_cast<float*>(_colors_buffer.data());

  // TODO this offset can be made part of each device,
  // and this entire function can pretty much be made generic as long as
  // the buffers are filled with the right types

  constexpr position rs2_offset{ 0.0f, 0.0f, 0.0f };

  // we take the negative positions of the Rs2 feed since they come in flipped
  // by default
  
  size_t point_count_out = 0;
  for (size_t i = 0; i < _point_count; i++) {
    const int pos = i * 3;
    float z_in = -_positions_buffer[pos + 2];
    if (config.flip_z) z_in *= -1;
    // // without any z value, it's an empty point, discard it
    if (z_in == 0) continue;
    // // without any color value, it's an empty point, discard it
    // auto color = colors_input[i];
    // if (color == std::numeric_limits<float>::min()) continue;
    // // check if it's within user-defined crop boundaries
    if (!config.crop_z.contains(z_in)) continue;
    float x_in = -_positions_buffer[pos];
    if (config.flip_x) x_in *= -1;
    if (!config.crop_x.contains(x_in)) continue;
    float y_in = -_positions_buffer[pos + 1];
    if (config.flip_y) y_in *= -1;
    if (!config.crop_y.contains(y_in)) continue;
    // apply device offset
    const float x_out = (x_in * config.scale) + rs2_offset.x + config.offset.x;
    const float y_out = (y_in * config.scale) + rs2_offset.y + config.offset.y;
    const float z_out = (z_in * config.scale) + rs2_offset.z + config.offset.z;
    // // add to our point cloud buffers
    positions[point_count_out] = (position{x_out, y_out, z_out, 0});
    // colors[point_count_out] = color;
    colors[point_count_out] = std::numeric_limits<float>::max();
    point_count_out++;
  }
  // resize buffers to the cropped count
  positions.resize(point_count_out);
  colors.resize(point_count_out);

  _point_cloud = PointCloud{positions, colors};
  _buffers_updated = false;

  return _point_cloud;
}

} // namespace bob::sensors
