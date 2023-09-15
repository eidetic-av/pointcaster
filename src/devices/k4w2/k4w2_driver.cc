#include "k4w2_driver.h"

namespace pc::devices {

using namespace pc::types;
using namespace Magnum; 

//libfreenect2::Freenect2 K4W2Driver::_freenect2;

K4W2Driver::K4W2Driver(int device_index_) {
  device_index = device_index_;
  open();
}

K4W2Driver::~K4W2Driver() { close(); }

bool K4W2Driver::open() {
    //_pipeline = new libfreenect2::OpenGLPacketPipeline();
    //pc::logger->info("dev count: {}", _freenect2.enumerateDevices());

 // // start a thread that captures new frames and dumps them into raw buffers
 // std::thread capture_loop([&]() {
 //   while (isOpen()) {
 //     k4a::capture capture;
 //     bool result = _device.get_capture(&capture, 1000ms);
 //     if (!result)
	//continue;
 //     auto depth_image = capture.get_depth_image();
 //     const int depth_width = depth_image.get_width_pixels();
 //     const int depth_height = depth_image.get_height_pixels();
 //     auto color_image = capture.get_color_image();

 //     // transform the color image to the depth image
 //     k4a::image transformed_color_image =
	//  _transformation.color_image_to_depth_camera(depth_image, color_image);
 //     // transform the depth image to a point cloud
 //     k4a::image point_cloud_image = _transformation.depth_image_to_point_cloud(
	//  depth_image, K4A_CALIBRATION_TYPE_DEPTH);

 //     auto colors_buffer = transformed_color_image.get_buffer();
 //     auto colors_buffer_size = transformed_color_image.get_size();

 //     auto positions_buffer =
	//  reinterpret_cast<int16_t *>(point_cloud_image.get_buffer());
 //     const auto positions_buffer_size = point_cloud_image.get_size();

 //     std::lock_guard<std::mutex> lock(_buffer_mutex);
 //     _point_count = depth_width * depth_height;
 //     _positions_buffer.resize(positions_buffer_size);
 //     memcpy(_positions_buffer.data(), positions_buffer, positions_buffer_size);
 //     _colors_buffer.resize(colors_buffer_size);
 //     memcpy(_colors_buffer.data(), colors_buffer, colors_buffer_size);
 //     _buffers_updated = true;
 //   }
 // });
 // capture_loop.detach();

  _open = true;
  return true;
}

bool K4W2Driver::close() {
  _open = false;
  return true;
}

PointCloud K4W2Driver::getPointCloud(const DeviceConfiguration& config) {
  if (!_buffers_updated)
    return _point_cloud;
  std::lock_guard<std::mutex> lock(_buffer_mutex);

  std::vector<position> positions(_point_count);
  std::vector<float> colors(_point_count);
  // memcpy(colors.data(), _colors_buffer.data(), _point_count * sizeof(float));
  auto colors_input = reinterpret_cast<float*>(_colors_buffer.data());

  constexpr position k4a_offset{ 0.0f, 1.0f, 0.0f };

  // TODO the following transform from short to float should be done in the
  // shader
  size_t point_count_out = 0;
  for (size_t i = 0; i < _point_count; i++) {
    const int pos = i * 3;
    float z_in = _positions_buffer[pos + 2] / -1000.0f;
    if (config.flip_z) z_in *= -1;
    // without any z value, it's an empty point, discard it
    if (z_in == 0) continue;
    // without any color value, it's an empty point, discard it
    auto color = colors_input[i];
    if (color == std::numeric_limits<float>::min()) continue;
    // check if it's within user-defined crop boundaries
    if (!config.crop_z.contains(z_in)) continue;
    float x_in = (_positions_buffer[pos] / -1000.0f);
    if (config.flip_x) x_in *= -1;
    if (!config.crop_x.contains(x_in)) continue;
    float y_in = (_positions_buffer[pos + 1] / -1000.0f);
    if (config.flip_y) y_in *= -1;
    if (!config.crop_y.contains(y_in)) continue;
    // apply device offset
    const float x_out = (x_in * config.scale) + k4a_offset.x + config.offset.x;
    const float y_out = (y_in * config.scale) + k4a_offset.y + config.offset.y;
    const float z_out = (z_in * config.scale) + k4a_offset.z + config.offset.z;
    // add to our point cloud buffers
    positions[point_count_out] = (position{x_out, y_out, z_out, 0});
    colors[point_count_out] = color;
    point_count_out++;
  }
  // resize buffers to the cropped count
  positions.resize(point_count_out);
  colors.resize(point_count_out);

  _point_cloud = PointCloud{positions, colors};
  _buffers_updated = false;

  return _point_cloud;
}

} // namespace pc::devices
