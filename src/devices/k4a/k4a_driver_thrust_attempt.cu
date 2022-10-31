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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

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
  _capture_loop = std::thread([&]() {
    while (!stop_requested) {
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
      positions_buffer.resize(positions_buffer_size);
      std::memcpy(positions_buffer.data(), point_cloud_image.get_buffer(),
		  positions_buffer_size);

      _buffers_updated = true;
    }
  });

  _open = true;
}

K4ADriver::~K4ADriver() {
  stop_requested = true;
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

	// move it into an Eigen object for easy maths
	aligned_orientation_offset = Eigen::Quaternion<float>(
	    pelvis_orientation.w, pelvis_orientation.x,
	    pelvis_orientation.y, pelvis_orientation.z);

	// get the euler angles
	const auto euler =
	    aligned_orientation_offset.toRotationMatrix().eulerAngles(0, 1, 2);

	// convert to degrees
	constexpr auto deg = [](float rad) -> float {
	  constexpr auto mult = 180.0f / 3.141592654f;
	  return rad * mult;
	};

        // align with kinect orientation offset
	const auto rot_x = deg(euler.x()) - 90;
	const auto rot_y = deg(euler.y()) - 90;
	const auto rot_z = deg(euler.z());

	spdlog::debug("0:{}, 1:{}, 2:{}", rot_x, rot_y, rot_z);
      }

      alignment_skeleton_frames.clear();
      aligned = true;
    }

  }
}

__global__
void transform(int point_count, short3 translation, float3 rotation,
	       short3 offset_position,
	       short3 *positions_in, short3 *positions_out) {
  using namespace Eigen;

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  // int index = 0; int stride = 1;
  for (int i = index; i < point_count; i += stride) {

    Vector3f pos;
    pos.x() = positions_in[i].x;
    pos.y() = positions_in[i].y;
    pos.z() = positions_in[i].z;

    pos.x() = pos.x();
    pos.y() = -pos.y();
    pos.z() = -pos.z();

    Vector3f rot_center { -translation.x, translation.y, -translation.z };

    Affine3f rot_x = Translation3f(rot_center) *
		     AngleAxisf(rotation.x, Vector3f::UnitX()) *
		     Translation3f(-rot_center);
    pos = rot_x * pos;

    Affine3f rot_y = Translation3f(rot_center) *
		     AngleAxisf(rotation.y, Vector3f::UnitY()) *
		     Translation3f(-rot_center);
    pos = rot_y * pos;

    Affine3f rot_z = Translation3f(rot_center) *
		     AngleAxisf(rotation.z, Vector3f::UnitZ()) *
		     Translation3f(-rot_center);
    pos = rot_z * pos;

    pos.x() = pos.x() + translation.x + offset_position.x;
    pos.y() = pos.y() + translation.y + offset_position.y;
    pos.z() = pos.z() + translation.z + offset_position.z;

    positions_out[i] = {
      (short) __float2int_rd(pos.x()),
      (short) __float2int_rd(pos.y()),
      (short) __float2int_rd(pos.z())
    };
  }
}

struct BoundsFilter {
  minMax<short> crop_x;
  minMax<short> crop_y;
  minMax<short> crop_z;

  __device__
  static bool contains(minMax<short> crop, short value) {
    if (value < crop.min) return false;
    if (value > crop.max) return false;
    return true;
  }

  __device__
  bool operator()(const short3 point) {
    // if (!contains(crop_z, point.z)) return false;
    // if (!contains(crop_x, point.x)) return false;
    // if (!contains(crop_y, point.y)) return false;
    return true;
  }
};

PointCloud K4ADriver::pointCloud(const DeviceConfiguration& config) {
  if (!_buffers_updated) return _point_cloud;
  std::lock_guard<std::mutex> lock(_buffer_mutex);

  // convert to radians
  const auto rad = [](float deg) -> float {
    constexpr auto mult = 3.141592654f / 180.0f;
    return deg * mult;
  };
  const float3 rotation_rad = {rad(config.rotation_deg.x),
			       rad(config.rotation_deg.y),
			       rad(config.rotation_deg.z)};

  _point_cloud.colors.resize(incoming_point_count);
  std::memcpy(_point_cloud.colors.data(), colors_buffer.data(),
	      incoming_point_count * sizeof(color));

  // init input and output GPU containers
  thrust::device_vector<short3> positions(positions_buffer);
  thrust::device_vector<short3> filtered_positions(incoming_point_count);

  // create a bounds filter with our configuration properties
  BoundsFilter within_bounds{config.crop_x, config.crop_y, config.crop_z};

  // copy the incoming points to our filtered_positions vector if they lie
  // within our crop bounds
  auto filtered_end =
    thrust::copy_if(thrust::device, positions.begin(), positions.end(),
		    filtered_positions.begin(), within_bounds);

  // spdlog::debug(filtered_positions.size());
  // filtered_positions.erase(filtered_end, filtered_positions.end());

  // // move our filtered_positions back to the CPU
  // _point_cloud.positions.resize(filtered_positions.size());
  // thrust::copy(filtered_positions.begin(), filtered_positions.end(),
  // 	       _point_cloud.positions.data());

  // spdlog::debug(filtered_positions.size());

  // auto positions_size = sizeof(short3) * incoming_point_count;
  
  // short3* origin_positions;
  // short3* transformed_positions;

  // cudaMallocManaged(&origin_positions, positions_size);
  // cudaMallocManaged(&transformed_positions, positions_size);

  // cudaMemcpy(origin_positions, positions_buffer.data(),
  // 	     positions_size, cudaMemcpyHostToDevice);

  // int blockSize = 256;
  // int numBlocks = (incoming_point_count + blockSize - 1) / blockSize;
  // transform<<<numBlocks, blockSize>>>(
  //     incoming_point_count, aligned_position_offset, rotation_rad,
  //     config.offset, origin_positions, transformed_positions);

  // cudaDeviceSynchronize();

  // _point_cloud.positions.resize(incoming_point_count);

  // cudaMemcpy(_point_cloud.positions.data(), transformed_positions,
  // 	     positions_size, cudaMemcpyDeviceToHost);

  // cudaFree(origin_positions);
  // cudaFree(transformed_positions);

  _buffers_updated = false;

  return _point_cloud;
}

} // namespace bob::sensors