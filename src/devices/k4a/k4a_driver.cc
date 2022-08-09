#include "k4a_driver.h"
#include <cstdint>
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <limits>
#include <spdlog/spdlog.h>
#include <yaclib/executor/thread_pool.hpp>
#include <yaclib/executor/strand.hpp>
#include <yaclib/executor/submit.hpp>

namespace bob::sensors {

using namespace bob::types;
using namespace Magnum;
using namespace std::chrono_literals;

K4ADriver::K4ADriver(int device_index_) {
  device_index = device_index_;
  open();
}

K4ADriver::~K4ADriver() { close(); }

bool K4ADriver::open() {
  _device = k4a::device::open(device_index);

  // TODO move the rest of this to the cpp api

  // TODO make config dynamic
  _config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
  _config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
  _config.color_resolution = K4A_COLOR_RESOLUTION_720P;
  /* _config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED; */
  _config.depth_mode = K4A_DEPTH_MODE_NFOV_2X2BINNED;
  _config.camera_fps = K4A_FRAMES_PER_SECOND_30;
  // _config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
  // _config.camera_fps = K4A_FRAMES_PER_SECOND_15;
  _config.synchronized_images_only = true;

  _device.start_cameras(&_config);

  // need to store the calibration and transformation data, as they are used
  // later to transform colour camera to point cloud space
  _calibration =
      _device.get_calibration(_config.depth_mode, _config.color_resolution);
  _transformation = k4a::transformation(_calibration);

  // start a thread that captures new frames and dumps them into raw buffers
  _capture_loop = std::jthread([&](std::stop_token st) {
    while (!st.stop_requested()) {
      k4a::capture capture;
      bool result = _device.get_capture(&capture, 1000ms);
      if (!result)
	continue;
      auto depth_image = capture.get_depth_image();
      const int depth_width = depth_image.get_width_pixels();
      const int depth_height = depth_image.get_height_pixels();
      auto color_image = capture.get_color_image();

      // transform the color image to the depth image
      k4a::image transformed_color_image =
	  _transformation.color_image_to_depth_camera(depth_image, color_image);
      // transform the depth image to a point cloud
      k4a::image point_cloud_image = _transformation.depth_image_to_point_cloud(
	  depth_image, K4A_CALIBRATION_TYPE_DEPTH);

      auto colors_buffer = transformed_color_image.get_buffer();
      auto colors_buffer_size = transformed_color_image.get_size();

      auto positions_buffer =
	  reinterpret_cast<short *>(point_cloud_image.get_buffer());
      const auto positions_buffer_size = point_cloud_image.get_size();

      std::lock_guard<std::mutex> lock(_buffer_mutex);
      _point_count = depth_width * depth_height;
      _positions_buffer.resize(positions_buffer_size);
      memcpy(_positions_buffer.data(), positions_buffer, positions_buffer_size);
      _colors_buffer.resize(colors_buffer_size);
      memcpy(_colors_buffer.data(), colors_buffer, colors_buffer_size);
      _buffers_updated = true;
    }
  });

  _open = true;
  return true;
}

bool K4ADriver::close() {
  _open = false;
  _device.close();
  return true;
}

auto tp = yaclib::MakeThreadPool(4);
auto strand = yaclib::MakeStrand(tp);

PointCloud K4ADriver::getPointCloud(const DeviceConfiguration& config) {
  if (!_buffers_updated) return _point_cloud;
  std::lock_guard<std::mutex> lock(_buffer_mutex);

  std::vector<short3> positions(_point_count);
  std::vector<color> colors(_point_count);
  auto colors_input = reinterpret_cast<color*>(_colors_buffer.data());

  constexpr float3 k4a_offset{ 0, 1.0, 0 };

  // TODO the following should be handled in a yaclib thread pool
  // compute shader, or p

  {
    // size_t point_count_out = 0;
    // spdlog::debug("Started serial");
    // for (std::size_t i = 0; i < _point_count; i++) {
    //   point_count_out++;
    // }

    // spdlog::debug("--> output {}", point_count_out);

    // point_count_out = 0;
    // spdlog::debug("Started parallel");
    // for (std::size_t i = 0; i < _point_count; i++) {
    //   Submit(*tp, [&] {
    // 	  point_count_out++;
    //   });
    // }
    // spdlog::debug("--> output {}", point_count_out);

  }

  size_t point_count_out = 0;
  for (size_t i = 0; i < _point_count; i++) {
    const int pos = i * 3;
    float z_in = _positions_buffer[pos + 2] / -1000.0f;
    if (config.flip_z) z_in *= -1;
    // without any z value, it's an empty point, discard it
    if (z_in == 0) continue;
    // without any color value, it's an empty point, discard it
    auto color = colors_input[i];

    // check if it's within user-defined crop boundaries
    if (!config.crop_z.contains(z_in)) continue;
    float x_in = (_positions_buffer[pos] / -1000.0f);
    if (config.flip_x) x_in *= -1;
    if (!config.crop_x.contains(x_in)) continue;
    float y_in = (_positions_buffer[pos + 1] / -1000.0f);
    if (config.flip_y) y_in *= -1;
    if (!config.crop_y.contains(y_in)) continue;
    // apply device offset
    const short x_out = ((x_in * config.scale) + k4a_offset.x + config.offset.x) * 1000;
    const short y_out = ((y_in * config.scale) + k4a_offset.y + config.offset.y) * 1000;
    const short z_out = ((z_in * config.scale) + k4a_offset.z + config.offset.z) * 1000;
    // add to our point cloud buffers
    positions[point_count_out] = {x_out, y_out, z_out};
    colors[point_count_out] = color;
    point_count_out++;
  }
  // resize buffers to the cropped count
  positions.resize(point_count_out);
  colors.resize(point_count_out);

  _point_cloud.positions = std::move(positions);
  _point_cloud.colors = std::move(colors);

  _buffers_updated = false;

  return _point_cloud;
}

} // namespace bob::sensors
