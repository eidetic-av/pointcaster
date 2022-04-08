#include "k4a_driver.h"
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>

namespace bob::sensors {

using namespace Magnum;

K4ADriver::K4ADriver(int _device_index) { device_index = _device_index; }

bool K4ADriver::Open() {
  if (k4a_device_open(device_index, &_device) != K4A_RESULT_SUCCEEDED)
    return false;

  _config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
  _config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
  // _config.color_resolution = K4A_COLOR_RESOLUTION_720P;
  _config.color_resolution = K4A_COLOR_RESOLUTION_OFF;
  _config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
  _config.camera_fps = K4A_FRAMES_PER_SECOND_30;

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

std::vector<Magnum::Vector3> K4ADriver::getPointCloud() {
  k4a_capture_t capture;
  constexpr int color_pixel_size = 4 * static_cast<int>(sizeof(uint8_t));
  constexpr int depth_pixel_size = static_cast<int>(sizeof(int16_t));
  constexpr int depth_point_size = 3 * static_cast<int>(sizeof(int16_t));

  const auto result = k4a_device_get_capture(_device, &capture, 1000);
  auto depth_image = k4a_capture_get_depth_image(capture);
  const int depth_width = k4a_image_get_width_pixels(depth_image);
  const int depth_height = k4a_image_get_height_pixels(depth_image);
  // auto color_image = k4a_capture_get_color_image(capture);

  // k4a_image_t transformed_color_image;
  // k4a_image_create(K4A_IMAGE_FORMAT_COLOR_BGRA32, depth_width, depth_height,
  // 		     depth_width * color_pixel_size, &transformed_color_image);
  k4a_image_t point_cloud_image;
  k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM, depth_width, depth_height,
		   depth_width * depth_point_size, &point_cloud_image);

  // // transform the color image to the depth image
  // k4a_transformation_color_image_to_depth_camera(kinect_transformation,
  // 						   depth_image, color_image,
  //                                                transformed_color_image);
  // // transform the depth image to a point cloud
  k4a_transformation_depth_image_to_point_cloud(_transformation, depth_image,
						K4A_CALIBRATION_TYPE_DEPTH,
						point_cloud_image);

  auto point_cloud_buffer =
      reinterpret_cast<int16_t *>(k4a_image_get_buffer(point_cloud_image));
  auto point_cloud_buffer_size = k4a_image_get_size(point_cloud_image);
  auto point_count = point_cloud_buffer_size / sizeof(int16_t) / 3;

  std::vector<Vector3> points;

  for (int i = 0; i < point_count; i++) {
    const int index = i * 3;
    const auto x = point_cloud_buffer[index] / -1000.f;
    const auto y = (point_cloud_buffer[index + 1] / -1000.f) + 1.f;
    const auto z = point_cloud_buffer[index + 2] / -1000.f;
    points.push_back(Vector3{x, y, z});
  }
  return points;
}

} // namespace bob::sensors
