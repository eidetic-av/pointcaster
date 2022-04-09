#include "k4a_driver.h"
#include <cstdint>
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <spdlog/spdlog.h>
#include <limits>

namespace bob::sensors {

using namespace Magnum;

K4ADriver::K4ADriver(int device_index_) { device_index = device_index_; }

bool K4ADriver::open() {
  if (k4a_device_open(device_index, &_device) != K4A_RESULT_SUCCEEDED)
    return false;

  // TODO make config dynamic
  _config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
  _config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
  _config.color_resolution = K4A_COLOR_RESOLUTION_720P;
  _config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
  _config.camera_fps = K4A_FRAMES_PER_SECOND_30;
  _config.synchronized_images_only = true;

  k4a_device_start_cameras(_device, &_config);

  // need to store the calibration and transformation data, as they are used
  // later to transform colour camera to point cloud space
  k4a_device_get_calibration(_device, _config.depth_mode,
			     _config.color_resolution, &_calibration);
  _transformation = k4a_transformation_create(&_calibration);

  // start a thread that captures new frames and dumps them into raw buffers
  std::thread capture_loop([&]() {

    constexpr int color_pixel_size = 4 * static_cast<int>(sizeof(uint8_t));
    constexpr int depth_pixel_size = static_cast<int>(sizeof(int16_t));
    constexpr int depth_point_size = 3 * static_cast<int>(sizeof(int16_t));

    while (isOpen()) {
      k4a_capture_t capture;

      const auto result = k4a_device_get_capture(_device, &capture, 1000);
      auto depth_image = k4a_capture_get_depth_image(capture);
      const int depth_width = k4a_image_get_width_pixels(depth_image);
      const int depth_height = k4a_image_get_height_pixels(depth_image);
      auto color_image = k4a_capture_get_color_image(capture);

      k4a_image_t transformed_color_image;
      k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32, depth_width, depth_height,
		       depth_width * color_pixel_size,
		       &transformed_color_image);

      k4a_image_t point_cloud_image;
      k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM, depth_width, depth_height,
		       depth_width * depth_point_size, &point_cloud_image);

      // transform the color image to the depth image
      k4a_transformation_color_image_to_depth_camera(
	  _transformation, depth_image, color_image, transformed_color_image);
      // transform the depth image to a point cloud
      k4a_transformation_depth_image_to_point_cloud(
	  _transformation, depth_image, K4A_CALIBRATION_TYPE_DEPTH,
	  point_cloud_image);

      auto positions_buffer =
	  reinterpret_cast<int16_t *>(k4a_image_get_buffer(point_cloud_image));
      const auto positions_buffer_size = k4a_image_get_size(point_cloud_image);

      auto colors_buffer = k4a_image_get_buffer(transformed_color_image);
      auto colors_buffer_size = k4a_image_get_size(transformed_color_image);

      _buffer_mutex.lock();
      _point_count = depth_width * depth_height;
      _positions_buffer.resize(positions_buffer_size);
      memcpy(_positions_buffer.data(), positions_buffer, positions_buffer_size);
      _colors_buffer.resize(colors_buffer_size);
      memcpy(_colors_buffer.data(), colors_buffer, colors_buffer_size);
      _buffer_mutex.unlock();

    }
  });
  capture_loop.detach();

  _open = true;
  return true;
}

bool K4ADriver::close() {
  _open = false;
  k4a_device_close(_device);
  return true;
}

PointCloud K4ADriver::getPointCloud() {
  _buffer_mutex.lock();

  std::vector<Vector3> positions(_point_count);
  std::vector<float> colors(_point_count);
  memcpy(colors.data(), _colors_buffer.data(), _point_count * sizeof(float));

  // TODO the following transform from short to float should be done in the
  // shader
  for (int i = 0; i < _point_count; i++) {
    const int pos = i * 3;
    const float x = (_positions_buffer[pos] / -1100.0f) + 1.25f;
    const float y = (_positions_buffer[pos + 1] / -1100.0f) + 0.55f;
    const float z = _positions_buffer[pos + 2] / -1100.0f;
    positions[i] = (Vector3{x, y, z});
  }

  _buffer_mutex.unlock();

  PointCloud points { positions, colors };
  return points;
}
} // namespace bob::sensors
