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
  spdlog::info("Beginning alignment for k4a {} ({})", device_index, id());
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
      
      // first average the joint positions over the captured frames
      const auto avg_joint_positions =
	calculateAverageJointPositions(alignment_skeleton_frames);

      // set the rotational centerto the position of the hips
      const auto left_hip_pos =
	avg_joint_positions[k4abt_joint_id_t::K4ABT_JOINT_HIP_LEFT];
      const auto right_hip_pos =
	avg_joint_positions[k4abt_joint_id_t::K4ABT_JOINT_HIP_RIGHT];
      const auto hip_x = (left_hip_pos.x + right_hip_pos.x) / 2.0f;
      const auto hip_y = (left_hip_pos.y + right_hip_pos.y) / 2.0f;
      const auto hip_z = (left_hip_pos.z + right_hip_pos.z) / 2.0f;

      rotation_center.x = -std::round(hip_x);
      rotation_center.y = std::round(hip_y);
      rotation_center.z = std::round(hip_z);

      // then set the y offset to the position of the feet
      const auto left_foot_pos =
	avg_joint_positions[k4abt_joint_id_t::K4ABT_JOINT_FOOT_LEFT];
      const auto right_foot_pos =
	avg_joint_positions[k4abt_joint_id_t::K4ABT_JOINT_FOOT_RIGHT];
      const auto feet_y = (left_foot_pos.y + right_foot_pos.y) / 2.0f;
      // (we do -hip_y + feet_y so that the origin (0,0) is aligned with the
      // floor and center of the skeleton, but our rotational center is the
      // hips because that's easier to reason about when making manual
      // adjustments)
      aligned_position_offset.y = std::round(-hip_y + feet_y);
      
      if (primary_aligner) {
	// if this is the 'primary' alignment device, we set the orientation
	// offset so that the skeleton faces 'forward' in the scene

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

    Vector3f rot_center { translation.x, translation.y, translation.z };
    AngleAxisf rot_z(rotation.z, Vector3f::UnitZ());
    AngleAxisf rot_y(rotation.y, Vector3f::UnitY());
    AngleAxisf rot_x(rotation.x, Vector3f::UnitX());
    Quaternionf q = rot_z * rot_y * rot_x;
    Affine3f rot_transform = Translation3f(-rot_center) * q * Translation3f(rot_center);
    pos = rot_transform * pos;

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

struct within_bounds {
  minMax<short> crop_x;
  minMax<short> crop_y;
  minMax<short> crop_z;

  __host__ __device__
  bool contains(minMax<short> crop, short value) const {
    if (value < crop.min) return false;
    if (value > crop.max) return false;
    return true;
  }

  __host__ __device__
  bool operator()(short3 pos) const {
    return contains(crop_x, pos.x) && contains(crop_y, pos.y) &&
	   contains(crop_z, pos.z);
  }
};

PointCloud
K4ADriver::pointCloud(const DeviceConfiguration &config) {
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
  const short3 position_offset = {config.offset.x + aligned_position_offset.x,
				  config.offset.y + aligned_position_offset.y,
				  config.offset.z + aligned_position_offset.z};

  const uint positions_size = sizeof(short3) * incoming_point_count;
  const uint colors_size = sizeof(color) * incoming_point_count;

  // initialize our GPU memory
  short3* incoming_positions;
  color* incoming_colors;
  short3* output_positions;
  color* output_colors;

  cudaMallocManaged(&incoming_positions, positions_size);
  cudaMallocManaged(&incoming_colors, colors_size);
  cudaMallocManaged(&output_positions, positions_size);
  cudaMallocManaged(&output_colors, colors_size);

  // fill the GPU memory with our buffers from the kinect on the CPU
  cudaMemcpy(incoming_positions, positions_buffer.data(),
	     positions_size, cudaMemcpyHostToDevice);
  cudaMemcpy(incoming_colors, colors_buffer.data(),
	     colors_size, cudaMemcpyHostToDevice);

  // make some thrust::device_ptr from the raw pointers so they work
  // <algorithm> style
  thrust::device_ptr<short3> incoming_positions_ptr =
    thrust::device_pointer_cast(incoming_positions);
  thrust::device_ptr<color> incoming_colors_ptr =
    thrust::device_pointer_cast(incoming_colors);
  thrust::device_ptr<short3> output_positions_ptr =
    thrust::device_pointer_cast(output_positions);
  thrust::device_ptr<color> output_colors_ptr =
    thrust::device_pointer_cast(output_colors);

  // zip position and color buffers together so we can run our algorithms on
  // both datasets as a single point-cloud
  auto point_cloud_begin = thrust::make_zip_iterator(
      thrust::make_tuple(incoming_positions_ptr, incoming_colors_pointer));
  auto point_cloud_end = thrust::make_zip_iterator(
      thrust::make_tuple(incoming_positions_ptr + incoming_point_count,
			 incoming_colors_pointer + incoming_point_count));

  // int rsize =
  //     thrust::copy_if(thrust::make_transform_iterator(
  //                         thrust::make_zip_iterator(thrust::make_tuple(
  //                             thrust::counting_iterator<int>(0), d_x.begin())),
  //                         add_random()),
  //                     thrust::make_transform_iterator(
  //                         thrust::make_zip_iterator(thrust::make_tuple(
  //                             thrust::counting_iterator<int>(5), d_x.end())),
  //                         add_random()),
  //                     d_r.begin(), is_greater()) -
  //     d_r.begin();

  // auto filtered_positions_end =
  //   thrust::copy_if(transformed_positions,
  // 		    transformed_positions + incoming_point_count,
  // 		    filtered_positions,
  // 		    within_bounds{ config.crop_x, config.crop_y, config.crop_z});

  // auto output_point_count = std::distance(filtered_positions, filtered_positions_end);
  // auto output_positions_size = sizeof(short3) * output_point_count;

  cudaDeviceSynchronize();

  // const uint output_positions_size = sizeof(short3) * output_point_count;
  // const uint output_colors_size = sizeof(color) * output_point_count;

  // // copy back to our output point-cloud on the CPU
  // _point_cloud.positions.resize(output_point_count);
  // cudaMemcpy(_point_cloud.positions.data(), output_positions,
  // 	     output_positions_size, cudaMemcpyDeviceToHost);
  // _point_cloud.colors.resize(output_point_count);
  // cudaMemcpy(_point_cloud.colors.data(), output_colors,
  // 	     output_colors_size, cudaMemcpyDeviceToHost);


  // clean up
  cudaFree(incoming_positions);
  cudaFree(incoming_colors);
  cudaFree(output_positions);
  cudaFree(output_colors);

  _buffers_updated = false;

  return _point_cloud;
}

} // namespace bob::sensors
