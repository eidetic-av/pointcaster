#include "rs2_driver.h"
#include <cstdint>
#include <librealsense2/hpp/rs_processing.hpp>
#include <limits>
#include <spdlog/spdlog.h>

namespace bob::sensors {

using namespace bob::types;
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
  _config.enable_stream(RS2_STREAM_COLOR, RS2_FORMAT_BGR8);
  _config.enable_stream(RS2_STREAM_DEPTH);
  auto profile = _pipe.start(_config);
  _device = profile.get_device();
  _serial_number = _device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
  _open = true;

  // start a thread that captures new frames and dumps them into raw buffers
  std::thread capture_loop([&]() {
    rs2::pointcloud point_cloud;
    rs2::points points;
    // wait until we open the device
    while (isOpen()) {
      // just block until we receive both depth and color data
      auto frames = _pipe.wait_for_frames();
      const rs2::depth_frame depth_frame = frames.get_depth_frame();
      const rs2::video_frame color_frame = frames.get_color_frame();

      // generate the vertex real-world positions from the depth_frame
      points = point_cloud.calculate(depth_frame);
      point_cloud.map_to(color_frame);
      // and the color texture coordinates
      auto texture = points.get_texture_coordinates();

      std::lock_guard<std::mutex> lock(_buffer_mutex);
      // move positions into buffer
      auto vertices = points.get_vertices();
      _point_count = depth_frame.get_width() * depth_frame.get_height();
      const size_t buffer_float_count = _point_count * 3;
      _positions_buffer.resize(buffer_float_count);
      std::memcpy(_positions_buffer.data(), vertices, buffer_float_count * sizeof(float));
      
      // get metadata needed to unpack colors from uvs
      _texture_width = color_frame.get_width();
      _texture_height = color_frame.get_height();
      _texture_pixel_size = color_frame.get_bytes_per_pixel();
      _texture_stride = color_frame.get_stride_in_bytes();
      auto pixel_count = _texture_width * _texture_height;
      // move colors into buffer
      _colors_buffer.resize(pixel_count * 3);
      std::memcpy(_colors_buffer.data(), color_frame.get_data(),
          pixel_count * sizeof(uint8_t) * 3);
      // move uv map into buffer
      _uvs_buffer.resize(_point_count);
      std::memcpy(_uvs_buffer.data(), texture,
          _point_count * sizeof(rs2::texture_coordinate));

      _buffers_updated = true;
    }
  });
  capture_loop.detach();

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

  std::vector<short3> positions(_point_count);
  std::vector<color> colors(_point_count);
  // memcpy(colors.data(), _colors_buffer.data(), _point_count * sizeof(float));
  // TODO this offset can be made part of each device,
  // and this entire function can pretty much be made generic as long as
  // the buffers are filled with the right types

  // we take the negative positions of the Rs2 feed since they come in flipped
  // by default
  
  size_t point_count_out = 0;
  for (size_t i = 0; i < _point_count; i++) {
    const int pos = i * 3;
    float z_in = -_positions_buffer[pos + 2];
    if (config.flip_z) z_in *= -1;
    // // without any z value, it's an empty point, discard it
    if (z_in == 0) continue;
    // // check if it's within user-defined crop boundaries
    if (!config.crop_z.contains(z_in)) continue;
    float x_in = -_positions_buffer[pos];
    if (config.flip_x) x_in *= -1;
    if (!config.crop_x.contains(x_in)) continue;
    float y_in = -_positions_buffer[pos + 1];
    if (config.flip_y) y_in *= -1;
    if (!config.crop_y.contains(y_in)) continue;
    // apply device offset
    const short x_out = ((x_in * config.scale) + config.offset.x) * 1000;
    const short y_out = ((y_in * config.scale) + config.offset.y) * 1000;
    const short z_out = ((z_in * config.scale) + config.offset.z) * 1000;
    // add position to our point cloud buffer
    positions[point_count_out] = short3{ x_out, y_out, z_out };

    // convert color from u/v to bgra8
    auto uv = _uvs_buffer[i];
    // normals to texture coordinates conversion
    int x_value = std::min(std::max(int(uv.u * _texture_width + .5f), 0), _texture_width - 1);
    int y_value = std::min(std::max(int(uv.v * _texture_height + .5f), 0), _texture_height - 1);
    int texture_index = (x_value * _texture_pixel_size) + (y_value * _texture_stride);
    // set the color now that we have the buffer index
    uint8_t color[] = {
        _colors_buffer[texture_index],
        _colors_buffer[texture_index + 1],
        _colors_buffer[texture_index + 2],
        std::numeric_limits<uint8_t>::max()
    };
    // and send it out as a bob::types::color value
    struct color packed;
    std::memcpy(&packed, &color, sizeof(float));
    colors[point_count_out] = packed;

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
