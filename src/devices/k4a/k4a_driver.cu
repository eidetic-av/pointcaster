#include "../../log.h"
#include "k4a_driver.h"
#include "k4a_utils.h"
#include <algorithm>
#include <cstdint>
#include <k4a/k4a.h>
#include <k4a/k4atypes.h>
#include <limits>
#include <numbers>
#include <numeric>
#include <system_error>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>

namespace pc::sensors {

using namespace pc::types;
using namespace pc::k4a_utils;
using namespace std::chrono_literals;
using namespace Magnum;
using namespace Magnum::Math;

uint K4ADriver::device_count = 0;

K4ADriver::K4ADriver() {

  device_index = device_count;

  pc::log.info("Opening driver for k4a %d (%s)", device_index, id());

  try {
    _device = k4a::device::open(device_index);
    _serial_number = _device.get_serialnum();
    pc::log.info("k4a %d Open", device_index);
  } catch (const std::system_error &e) {
    pc::log.error("Failed to open k4a::device due to %s", e.code());
    pc::log.error("%s", e.what());
    return;
  }

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

  pc::log.info("k4a %d attempting to start camera", device_index);
  try {
    _device.start_cameras(&_config);
    pc::log.info("k4a %d started camera", device_index);
  } catch (std::exception e) {
    pc::log.error("k4a %d failed to start camera", device_index);
    pc::log.error(e.what());
    return;
  }

  // need to store the calibration and transformation data, as they are used
  // later to transform colour camera to point cloud space
  _calibration =
      _device.get_calibration(_config.depth_mode, _config.color_resolution);
  _transformation = k4a::transformation(_calibration);

  _tracker = k4abt::tracker::create(_calibration);

  // start a thread that captures new frames and dumps them into raw buffers
  _capture_loop = std::thread([&]() {
    while (!_stop_requested) {

      if (_pause_sensor) {
        std::this_thread::sleep_for(100ms);
        continue;
      }

      k4a::capture capture;
      bool result = _device.get_capture(&capture, 1000ms);
      if (!result)
        continue;

      if (is_aligning())
        run_aligner(capture);

      auto depth_image = capture.get_depth_image();
      auto color_image = capture.get_color_image();

      k4a::image transformed_color_image =
          _transformation.color_image_to_depth_camera(depth_image, color_image);
      k4a::image point_cloud_image = _transformation.depth_image_to_point_cloud(
          depth_image, K4A_CALIBRATION_TYPE_DEPTH);

      std::lock_guard<std::mutex> lock(_buffer_mutex);

      std::memcpy(_colors_buffer.data(), transformed_color_image.get_buffer(),
                  color_buffer_size);
      std::memcpy(_positions_buffer.data(), point_cloud_image.get_buffer(),
                  positions_buffer_size);

      _buffers_updated = true;
    }
  });

  _open = true;

  device_count++;
}

K4ADriver::~K4ADriver() {
  pc::log.info("Closing driver for k4a %d (%s)", device_index, id());
  _open = false;
  _stop_requested = true;
  _capture_loop.join();
  _tracker.destroy();
  _device.stop_cameras();
  _device.close();
  pc::log.info("k4a %d driver closed", device_index);
  device_count--;
}

bool K4ADriver::is_open() const { return _open; }

std::string K4ADriver::id() const { return _serial_number; }

void K4ADriver::set_paused(bool paused) { _pause_sensor = paused; }

void K4ADriver::start_alignment() {
  pc::log.info("Beginning alignment for k4a %d (%s)", device_index, id());
  _alignment_frame_count = 0;
}

bool K4ADriver::is_aligning() {
  return _alignment_frame_count < _total_alignment_frames;
}

bool K4ADriver::is_aligned() { return _aligned; }

void K4ADriver::run_aligner(const k4a::capture &frame) {
  _tracker.enqueue_capture(frame);
  k4abt::frame body_frame = _tracker.pop_result();
  const auto body_count = body_frame.get_num_bodies();

  if (body_count > 0) {
    const k4abt_skeleton_t skeleton = body_frame.get_body(0).skeleton;
    _alignment_skeleton_frames.push_back(skeleton);

    if (++_alignment_frame_count == _total_alignment_frames) {
      // alignment frames have finished capturing

      // first average the joint positions over the captured frames
      const auto avg_joint_positions =
          calculateAverageJointPositions(_alignment_skeleton_frames);

      // set the rotational centerto the position of the hips
      const auto left_hip_pos =
          avg_joint_positions[k4abt_joint_id_t::K4ABT_JOINT_HIP_LEFT];
      const auto right_hip_pos =
          avg_joint_positions[k4abt_joint_id_t::K4ABT_JOINT_HIP_RIGHT];
      const auto hip_x = (left_hip_pos.x + right_hip_pos.x) / 2.0f;
      const auto hip_y = (left_hip_pos.y + right_hip_pos.y) / 2.0f;
      const auto hip_z = (left_hip_pos.z + right_hip_pos.z) / 2.0f;

      _alignment_center.x = -std::round(hip_x);
      _alignment_center.y = std::round(hip_y);
      _alignment_center.z = std::round(hip_z);

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
      _aligned_position_offset.y = std::round(-hip_y + feet_y);

      if (primary_aligner) {
        // if this is the 'primary' alignment device, we set the orientation
        // offset so that the skeleton faces 'forward' in the scene

        // average the joint orientations over the captured frames
        const auto avg_joint_orientations =
            calculateAverageJointOrientations(_alignment_skeleton_frames);
        // assuming the pelvis is the source of truth for orientation
        const auto pelvis_orientation =
            avg_joint_orientations[k4abt_joint_id_t::K4ABT_JOINT_PELVIS];

        // move it into an Eigen object for easy maths
        _aligned_orientation_offset = Eigen::Quaternion<float>(
            pelvis_orientation.w, pelvis_orientation.x, pelvis_orientation.y,
            pelvis_orientation.z);

        // get the euler angles
        const auto euler =
            _aligned_orientation_offset.toRotationMatrix().eulerAngles(0, 1, 2);

        // convert to degrees
        constexpr auto deg = [](float rad) -> float {
          constexpr auto mult = 180.0f / 3.141592654f;
          return rad * mult;
        };

        // align with kinect orientation offset
        const auto rot_x = deg(euler.x()) - 90;
        const auto rot_y = deg(euler.y()) - 90;
        const auto rot_z = deg(euler.z());
      }

      _alignment_skeleton_frames.clear();
      _aligned = true;
    }
  }
}

__host__ __device__ static inline float rad(float deg) {
  constexpr auto mult = 3.141592654f / 180.0f;
  return deg * mult;
};

// TODO these GPU kernels can probably be taken outside of the k4a classes and
// used with any sensor type

typedef thrust::tuple<short3, color, int> point_in_t;
typedef thrust::tuple<position, color> point_out_t;

struct point_transformer
    : public thrust::unary_function<point_in_t, point_out_t> {

  DeviceConfiguration config;
  Eigen::Vector3f alignment_center;
  Eigen::Vector3f aligned_position_offset;

  point_transformer(const DeviceConfiguration &device_config,
                    const position &aligned_center,
                    const position &position_offset) {
    config = device_config;
    alignment_center =
        Eigen::Vector3f(aligned_center.x, aligned_center.y, aligned_center.z);
    aligned_position_offset = Eigen::Vector3f(
        position_offset.x, position_offset.y, position_offset.z);
  }

  __device__ point_out_t operator()(point_in_t point) const {

    // we reinterpret our point in Eigen containers so we have easy maths
    using namespace Eigen;

    short3 pos = thrust::get<0>(point);

    // we put our position into a float vector because it allows us to
    // transform it by other float types (e.g. matrices, quaternions)
    // -- also flip the y value to move kinect coords into the right
    // orientation for our scene
    Vector3f pos_f(pos.x, -pos.y, -pos.z);

    // create the rotation around our center
    AngleAxisf rot_z(rad(config.rotation_deg.z), Vector3f::UnitZ());
    AngleAxisf rot_y(rad(config.rotation_deg.y), Vector3f::UnitY());
    AngleAxisf rot_x(rad(config.rotation_deg.x), Vector3f::UnitX());
    Quaternionf q = rot_z * rot_y * rot_x;
    Affine3f rot_transform =
        Translation3f(-alignment_center) * q * Translation3f(alignment_center);

    // perform our initial transform
    pos_f =
        (rot_transform * pos_f) + alignment_center + aligned_position_offset;

    // perform our manual translation
    pos_f += Vector3f(config.offset.x, config.offset.y, config.offset.z);

    // specified axis flips
    if (config.flip_x)
      pos_f.x() = -pos_f.x();
    if (config.flip_y)
      pos_f.y() = -pos_f.y();
    if (config.flip_z)
      pos_f.z() = -pos_f.z();

    // and scaling
    pos_f *= config.scale;

    position pos_out = {(short)__float2int_rd(pos_f.x()),
                        (short)__float2int_rd(pos_f.y()),
                        (short)__float2int_rd(pos_f.z()), 0};

    color col = thrust::get<1>(point);
    // TODO apply color transformations here

    return thrust::make_tuple(pos_out, col);
  }
};

struct point_filter {
  DeviceConfiguration config;

  __device__ bool check_color(color value) const {
    // remove totally black values
    if (value.r == 0 && value.g == 0 && value.b == 0)
      return false;
    return true;
  }

  __device__ bool check_bounds(short3 value) const {
    if (value.x < config.crop_x.min)
      return false;
    if (value.x > config.crop_x.max)
      return false;
    if (value.y < config.crop_y.min)
      return false;
    if (value.y > config.crop_y.max)
      return false;
    if (value.z < config.crop_z.min)
      return false;
    if (value.z > config.crop_z.max)
      return false;
    return true;
  }

  __device__ bool sample(int index) const { return index % config.sample == 0; }

  __device__ bool operator()(point_in_t point) const {
    auto index = thrust::get<2>(point);
    if (!sample(index))
      return false;
    auto color = thrust::get<1>(point);
    if (!check_color(color))
      return false;
    auto position = thrust::get<0>(point);
    if (!check_bounds(position))
      return false;
    return true;
  }
};

using thrust_point_t = thrust::tuple<position, color>;

struct Translate {
  float x;
  float y;
  float z;

  __device__ thrust_point_t operator()(thrust_point_t point) const {
    auto position = thrust::get<0>(point);
    position.x += x;
    position.y += y;
    position.z += z;
    return thrust::make_tuple(position, thrust::get<1>(point));
  }
};

struct Scale {
  float x;
  float y;
  float z;

  __device__ thrust_point_t operator()(thrust_point_t point) const {
    auto position = thrust::get<0>(point);
    position.x *= x;
    position.y *= y;
    position.z *= z;
    return thrust::make_tuple(position, thrust::get<1>(point));
  }
};

template <typename Iterator, typename... Transformations>
auto make_transform_pipeline(Iterator begin, size_t count,
                             Transformations &&...transformations) {
  auto it = thrust::make_transform_iterator(
      begin, std::forward<Transformations>(transformations)...);
  auto it_end = thrust::make_transform_iterator(
      begin + count, std::forward<Transformations>(transformations)...);
  return std::make_pair<Iterator, Iterator>(it, it_end);
  // thrust::transform(it, it_end, begin,
  // std::forward<Transformations>(transformations)...);
}

PointCloud K4ADriver::point_cloud(const DeviceConfiguration &config) {
  if (!_buffers_updated || !is_open())
    return _point_cloud;

  std::lock_guard<std::mutex> lock(_buffer_mutex);

  const uint positions_in_size = sizeof(short3) * incoming_point_count;
  const uint positions_out_size = sizeof(position) * incoming_point_count;
  const uint colors_size = sizeof(color) * incoming_point_count;

  // initialize our GPU memory
  short3 *incoming_positions;
  color *incoming_colors;
  short3 *filtered_positions;
  color *filtered_colors;
  position *output_positions;
  color *output_colors;

  cudaMallocManaged(&incoming_positions, positions_in_size);
  cudaMallocManaged(&incoming_colors, colors_size);
  cudaMallocManaged(&filtered_positions, positions_in_size);
  cudaMallocManaged(&filtered_colors, colors_size);
  cudaMallocManaged(&output_positions, positions_out_size);
  cudaMallocManaged(&output_colors, colors_size);

  // fill the GPU memory with our CPU buffers from the kinect
  cudaMemcpy(incoming_positions, _positions_buffer.data(), positions_in_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(incoming_colors, _colors_buffer.data(), colors_size,
             cudaMemcpyHostToDevice);

  // make some thrust::device_ptr from the raw pointers so they work
  // <algorithm> style
  thrust::device_ptr<short3> incoming_positions_ptr =
      thrust::device_pointer_cast(incoming_positions);
  thrust::device_ptr<color> incoming_colors_ptr =
      thrust::device_pointer_cast(incoming_colors);
  thrust::device_ptr<short3> filtered_positions_ptr =
      thrust::device_pointer_cast(filtered_positions);
  thrust::device_ptr<color> filtered_colors_ptr =
      thrust::device_pointer_cast(filtered_colors);
  thrust::device_ptr<position> output_positions_ptr =
      thrust::device_pointer_cast(output_positions);
  thrust::device_ptr<color> output_colors_ptr =
      thrust::device_pointer_cast(output_colors);

  // we create some sequence of point indices to zip so our GPU kernels have
  // access to them
  int *indices;
  cudaMallocManaged(&indices, positions_in_size);
  thrust::device_ptr<int> point_indices = thrust::device_pointer_cast(indices);
  thrust::sequence(point_indices, point_indices + incoming_point_count);

  // zip position and color buffers together so we can run our algorithms on
  // the dataset as a single point-cloud
  auto incoming_points_begin = thrust::make_zip_iterator(thrust::make_tuple(
      incoming_positions_ptr, incoming_colors_ptr, point_indices));
  auto incoming_points_end = thrust::make_zip_iterator(
      thrust::make_tuple(incoming_positions_ptr + incoming_point_count,
                         incoming_colors_ptr + incoming_point_count,
                         point_indices + incoming_point_count));
  auto filtered_points_begin = thrust::make_zip_iterator(thrust::make_tuple(
      filtered_positions_ptr, filtered_colors_ptr, point_indices));
  auto output_points_begin = thrust::make_zip_iterator(
      thrust::make_tuple(output_positions_ptr, output_colors_ptr));

  // run the kernels
  auto filtered_points_end =
      thrust::copy_if(incoming_points_begin, incoming_points_end,
                      filtered_points_begin, point_filter{config});
  thrust::transform(
      filtered_points_begin, filtered_points_end, output_points_begin,
      point_transformer(config, _alignment_center, _aligned_position_offset));

  // we can determine the output count using the resulting output iterator
  // from running the kernels
  auto output_point_count =
      std::distance(filtered_points_begin, filtered_points_end);

  // auto [it, it_end] = make_transform_pipeline(output_points_begin,
  // output_point_count,
  //                    Translate{1000, 1000, 1000});

  // TRANSLATE
  // auto [transform_start, transform_end] = make_transform_pipeline(
  //     output_points_begin, output_points_begin + output_point_count,
  //     Translate{5000, 5000, 5000});

  // thrust::transform(translate_start,
  // 		    translate_end,
  // 		    output_points_begin, Translate{-5000, -5000, -5000});

  // wait for the GPU process to complete
  cudaDeviceSynchronize();

  // copy back to our output point-cloud on the CPU
  const uint output_positions_size = sizeof(position) * output_point_count;
  const uint output_colors_size = sizeof(color) * output_point_count;
  _point_cloud.positions.resize(output_point_count);
  _point_cloud.colors.resize(output_point_count);
  cudaMemcpy(_point_cloud.positions.data(), output_positions,
             output_positions_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(_point_cloud.colors.data(), output_colors, output_colors_size,
             cudaMemcpyDeviceToHost);

  // clean up
  cudaFree(indices);
  cudaFree(incoming_positions);
  cudaFree(incoming_colors);
  cudaFree(filtered_positions);
  cudaFree(filtered_colors);
  cudaFree(output_positions);
  cudaFree(output_colors);

  _buffers_updated = false;

  return _point_cloud;
}

void K4ADriver::set_exposure(const int new_exposure) {
  _device.set_color_control(K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                            K4A_COLOR_CONTROL_MODE_MANUAL, new_exposure);
}

int K4ADriver::get_exposure() const {
  int result_value = -1;
  k4a_color_control_mode_t result_mode;
  _device.get_color_control(K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                            &result_mode, &result_value);
  return result_value;
}

void K4ADriver::set_brightness(const int new_brightness) {
  _device.set_color_control(K4A_COLOR_CONTROL_BRIGHTNESS,
                            K4A_COLOR_CONTROL_MODE_MANUAL, new_brightness);
}

int K4ADriver::get_brightness() const {
  int result_value = -1;
  k4a_color_control_mode_t result_mode;
  _device.get_color_control(K4A_COLOR_CONTROL_BRIGHTNESS, &result_mode,
                            &result_value);
  return result_value;
}

void K4ADriver::set_contrast(const int new_contrast) {
  _device.set_color_control(K4A_COLOR_CONTROL_CONTRAST,
                            K4A_COLOR_CONTROL_MODE_MANUAL, new_contrast);
}

int K4ADriver::get_contrast() const {
  int result_value = -1;
  k4a_color_control_mode_t result_mode;
  _device.get_color_control(K4A_COLOR_CONTROL_CONTRAST, &result_mode,
                            &result_value);
  return result_value;
}

void K4ADriver::set_saturation(const int new_saturation) {
  _device.set_color_control(K4A_COLOR_CONTROL_SATURATION,
                            K4A_COLOR_CONTROL_MODE_MANUAL, new_saturation);
}

int K4ADriver::get_saturation() const {
  int result_value = -1;
  k4a_color_control_mode_t result_mode;
  _device.get_color_control(K4A_COLOR_CONTROL_SATURATION, &result_mode,
                            &result_value);
  return result_value;
}

void K4ADriver::set_gain(const int new_gain) {
  _device.set_color_control(K4A_COLOR_CONTROL_GAIN,
                            K4A_COLOR_CONTROL_MODE_MANUAL, new_gain);
}

int K4ADriver::get_gain() const {
  int result_value = -1;
  k4a_color_control_mode_t result_mode;
  _device.get_color_control(K4A_COLOR_CONTROL_GAIN, &result_mode,
                            &result_value);
  return result_value;
}

} // namespace pc::sensors
