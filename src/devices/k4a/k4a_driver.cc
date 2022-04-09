#include "k4a_driver.h"
#include <cstdint>
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <spdlog/spdlog.h>
#include <limits>

namespace bob::sensors {

using namespace Magnum;

K4ADriver::K4ADriver(int _device_index) { device_index = _device_index; }

bool K4ADriver::Open() {
  if (k4a_device_open(device_index, &_device) != K4A_RESULT_SUCCEEDED)
    return false;

  _config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
  _config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
  _config.color_resolution = K4A_COLOR_RESOLUTION_720P;
  // _config.color_resolution = K4A_COLOR_RESOLUTION_OFF;
  _config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
  _config.camera_fps = K4A_FRAMES_PER_SECOND_30;

  _config.synchronized_images_only = true;

  k4a_device_start_cameras(_device, &_config);

  // set up calibration
  k4a_device_get_calibration(_device, _config.depth_mode,
			     _config.color_resolution, &_calibration);
  _transformation = k4a_transformation_create(&_calibration);

  _open = true;
  return true;
}

bool K4ADriver::Close() {
  k4a_device_close(_device);
  return true;
}


PointCloud<Vector3, float> K4ADriver::getPointCloud() {
  k4a_capture_t capture;
  constexpr int color_pixel_size = 4 * static_cast<int>(sizeof(uint8_t));
  constexpr int depth_pixel_size = static_cast<int>(sizeof(int16_t));
  constexpr int depth_point_size = 3 * static_cast<int>(sizeof(int16_t));

  const auto result = k4a_device_get_capture(_device, &capture, 1000);
  auto depth_image = k4a_capture_get_depth_image(capture);
  const int depth_width = k4a_image_get_width_pixels(depth_image);
  const int depth_height = k4a_image_get_height_pixels(depth_image);
  auto color_image = k4a_capture_get_color_image(capture);
  // if (color_image == nullptr) return std::vector<Magnum::Vector3> { } ;

  k4a_image_t transformed_color_image;
  k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32, depth_width, depth_height,
		   depth_width * color_pixel_size, &transformed_color_image);

  k4a_image_t point_cloud_image;
  k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM, depth_width, depth_height,
		   depth_width * depth_point_size, &point_cloud_image);

  // transform the color image to the depth image
  k4a_transformation_color_image_to_depth_camera(
      _transformation, depth_image, color_image, transformed_color_image);
  // transform the depth image to a point cloud
  k4a_transformation_depth_image_to_point_cloud(_transformation, depth_image,
						K4A_CALIBRATION_TYPE_DEPTH,
						point_cloud_image);

  auto positions_buffer =
      reinterpret_cast<int16_t *>(k4a_image_get_buffer(point_cloud_image));
  auto positions_buffer_size = k4a_image_get_size(point_cloud_image);

  auto colors_buffer = k4a_image_get_buffer(transformed_color_image);
  auto colors_buffer_size = depth_width * depth_height * sizeof(float);

  auto point_count = depth_width * depth_height;

  std::vector<Vector3> positions(point_count);
  std::vector<float> colors(point_count);
  memcpy(colors.data(), colors_buffer, point_count * sizeof(float));
  // spdlog::info("colors size: ", colors_buffer_size);

  for (int i = 0; i < point_count; i++) {
    const int pos = i * 3;
    const auto x = positions_buffer[pos] / -1000.f;
    const auto y = (positions_buffer[pos + 1] / -1000.f) + 1.f;
    const auto z = positions_buffer[pos + 2] / -1000.f;
    positions[i] = (Vector3{x, y, z});

    // const int col = i * 4;
    // const auto r = ((float) colors_buffer[col + 2]) / std::numeric_limits<uint8_t>::max();
    // const auto g = ((float) colors_buffer[col + 1]) / std::numeric_limits<uint8_t>::max();
    // const auto b = ((float) colors_buffer[col]) / std::numeric_limits<uint8_t>::max();
    // spdlog::debug("r: {}, g: {}, b: {}", r, g, b);
    // colors.push_back(Vector4{1, 0, 1, 1});
    // colors.push_back(std::numeric_limits<float>::min());
  }

  PointCloud<Vector3, float> points { positions, colors };
  // spdlog::info("ps: {}", points.positions.size());
  return points;
}

} // namespace bob::sensors
