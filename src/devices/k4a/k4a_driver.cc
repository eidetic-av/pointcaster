#include "k4a_driver.h"
#include "k4a_utils.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <numbers>
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <spdlog/spdlog.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

namespace bob::sensors {

using namespace bob::types;
using namespace bob::k4a_utils;
using namespace std::chrono_literals;
using namespace Magnum;
using namespace Magnum::Math;

K4ADriver::K4ADriver(int device_index_) {
  device_index = device_index_;

  spdlog::info("Opening k4a device at index {}", device_index);
  device = k4a::device::open(device_index);
  serial_number = device.get_serialnum();
  spdlog::info(" --> Open");

  // TODO make config dynamic
  _config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
  _config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
  _config.color_resolution = K4A_COLOR_RESOLUTION_720P;
  /* _config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED; */
  _config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
  _config.camera_fps = K4A_FRAMES_PER_SECOND_30;
  // _config.depth_mode = K4A_DEPTH_MODE_WFOV_UNBINNED;
  // _config.camera_fps = K4A_FRAMES_PER_SECOND_15;
  _config.synchronized_images_only = true;

  spdlog::info("Attempting to start camera");
  device.start_cameras(&_config);
  spdlog::info("Started camera");

  // need to store the calibration and transformation data, as they are used
  // later to transform colour camera to point cloud space
  _calibration =
      device.get_calibration(_config.depth_mode, _config.color_resolution);
  _transformation = k4a::transformation(_calibration);

  tracker = k4abt::tracker::create(_calibration);

  // start a thread that captures new frames and dumps them into raw buffers
  _capture_loop = std::jthread([&](std::stop_token st) {
    while (!st.stop_requested()) {
      k4a::capture capture;
      bool result = device.get_capture(&capture, 1000ms);
      if (!result) continue;

      if (isAligning()) runAligner(capture);

      auto depth_image = capture.get_depth_image();
      auto color_image = capture.get_color_image();

      k4a::image transformed_color_image =
	  _transformation.color_image_to_depth_camera(depth_image, color_image);
      k4a::image point_cloud_image = _transformation.depth_image_to_point_cloud(
	  depth_image, K4A_CALIBRATION_TYPE_DEPTH);

      std::lock_guard<std::mutex> lock(_buffer_mutex);

      std::memcpy(colors_buffer.data(), transformed_color_image.get_buffer(),
		  color_buffer_size);
      std::memcpy(positions_buffer.data(), point_cloud_image.get_buffer(),
		  positions_buffer_size);

      _buffers_updated = true;
    }
  });

  _open = true;
}

K4ADriver::~K4ADriver() {
  _capture_loop.request_stop();
  _capture_loop.join();
  tracker.destroy();
  _open = false;
  device.close();
}

bool K4ADriver::isOpen() const { return _open; }

std::string K4ADriver::id() const { return serial_number; }

void K4ADriver::startAlignment() {
  alignment_frame_count = 0;
}

bool K4ADriver::isAligning() {
  return alignment_frame_count < total_alignment_frames;
}

bool K4ADriver::isAligned() {
  return aligned;
}

void K4ADriver::runAligner(const k4a::capture &frame) {
  tracker.enqueue_capture(frame);
  k4abt::frame body_frame = tracker.pop_result();
  const auto body_count = body_frame.get_num_bodies();

  if (body_count > 0) {
    const k4abt_skeleton_t skeleton = body_frame.get_body(0).skeleton;
    alignment_skeleton_frames.push_back(skeleton);

    if (++alignment_frame_count == total_alignment_frames) {
      // alignment frames have finished capturing
      if (primary_aligner) {
	// if this is the 'primary' alignment device, we set the position and
	// orientation offset so that the skeleton is centred in the scene

	// first average the joint positions over the captured frames
        const auto avg_joint_positions =
            calculateAverageJointPositions(alignment_skeleton_frames);

        // then set the y offset to the position of the feet
        const auto left_foot_pos =
            avg_joint_positions[k4abt_joint_id_t::K4ABT_JOINT_FOOT_LEFT];
        const auto right_foot_pos =
            avg_joint_positions[k4abt_joint_id_t::K4ABT_JOINT_FOOT_RIGHT];
        const auto feet_y = (left_foot_pos.y + right_foot_pos.y) / 2.0f;
        // set the x and z offset to the position of the hips
        const auto left_hip_pos =
            avg_joint_positions[k4abt_joint_id_t::K4ABT_JOINT_HIP_LEFT];
        const auto right_hip_pos =
            avg_joint_positions[k4abt_joint_id_t::K4ABT_JOINT_HIP_RIGHT];
        const auto hip_x = (left_hip_pos.x + right_hip_pos.x) / 2.0f;
        const auto hip_z = (left_hip_pos.z + right_hip_pos.z) / 2.0f;

	aligned_position_offset.x = -std::round(hip_x);
	aligned_position_offset.y = std::round(feet_y);
	aligned_position_offset.z = std::round(hip_z);

        // average the joint orientations over the captured frames
        const auto avg_joint_orientations =
            calculateAverageJointOrientations(alignment_skeleton_frames);
        // assuming the pelvis is the source of truth for orientation
	const auto pelvis_orientation =
	    avg_joint_orientations[k4abt_joint_id_t::K4ABT_JOINT_PELVIS];

	aligned_orientation_offset = Eigen::Quaternion<float>(
	    pelvis_orientation.w, pelvis_orientation.x,
	    pelvis_orientation.y, pelvis_orientation.z);

        // const auto pelvis_quat =
	//     Quaternion({pelvis_orientation.x, pelvis_orientation.y,
	// 		pelvis_orientation.z}, pelvis_orientation.w);
	// // add the camera angle offset for each axis
	// constexpr auto angle_offset = Magnum::Math::Rad<float>(90 * std::numbers::pi / 180);
	// const auto euler = pelvis_quat.toEuler();
	// const auto x = euler.x() + angle_offset;
	// const auto y = euler.y();
	// const auto z = euler.z() + angle_offset;

	// TODO not sure why these need to be doubled... is it correct?
	// TODO figure out how to create a rotation about more than one axis
	// aligned_orientation_offset =
	    // Quaternion<float>::rotation(-y, Vector3<float>::xAxis()) *
	    // Quaternion<float>::rotation(z, Vector3<float>::zAxis()) *
	    // Quaternion<float>::rotation(-x * 2, Vector3<float>::yAxis());

	// const auto e = aligned_orientation_offset.toEuler();
        // const auto x_e = (float) e.x() * 180 / std::numbers::pi;
	// const auto y_e = (float) e.y() * 180 / std::numbers::pi;
	// const auto z_e = (float) e.z() * 180 / std::numbers::pi;
	// spdlog::debug("x: {}, y: {}, z: {}", x_e, y_e, z_e);

      }

      alignment_skeleton_frames.clear();
      aligned = true;
    }

  }
}

PointCloud K4ADriver::pointCloud(const DeviceConfiguration& config) {
  if (!_buffers_updated) return _point_cloud;
  std::lock_guard<std::mutex> lock(_buffer_mutex);

  std::vector<short3> positions(incoming_point_count);
  std::vector<color> colors(incoming_point_count);

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.resize(incoming_point_count);

  size_t point_count_out = 0;
  for (size_t i = 0; i < incoming_point_count; i++) {
    Eigen::Vector3<float> pos;
    auto& x = pos.data()[0];
    auto& y = pos.data()[1];
    auto& z = pos.data()[2];

    z = positions_buffer[i].z * -1;
    // without any z value, it's an empty point, discard it
    if (z == 0) continue;

    if (config.flip_z) z *= -1;
    if (!config.crop_z.contains(z)) continue;
    x = positions_buffer[i].x;
    if (config.flip_x) x *= -1;
    if (!config.crop_x.contains(x)) continue;
    y = positions_buffer[i].y * -1;
    if (config.flip_y) y *= -1;
    if (!config.crop_y.contains(y)) continue;

    // apply alignment transformations
    if (aligned) {
      x += aligned_position_offset.x;
      y += aligned_position_offset.y;
      z += aligned_position_offset.z;

      // pos = rotate(pos, aligned_orientation_offset);
      pos = aligned_orientation_offset * pos;
    }

    // const short x_out = (x_in * config.scale) + config.offset.x;
    // const short y_out = (y_in * config.scale) + config.offset.y;
    // const short z_out = (z_in * config.scale) + config.offset.z;

    // add to our point cloud buffers
    positions[point_count_out] = { (short) x, (short) y, (short) z };
    colors[point_count_out] = colors_buffer.data()[i];
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
