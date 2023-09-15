#include "k4a_driver.h"
#include "../../logger.h"
#include "k4a_utils.h"
#include <future>
#include <numeric>

namespace pc::devices {

using namespace std::chrono_literals;
using namespace pc::k4a_utils;

uint K4ADriver::device_count = 0;

K4ADriver::K4ADriver(const DeviceConfiguration &config) {
  static std::mutex serial_driver_construction;
  std::lock_guard<std::mutex> lock(serial_driver_construction);

  device_index = device_count;

  pc::logger->info("Opening driver for k4a {} ({})\n", device_index, id());

  _device = k4a::device::open(device_index);
  _serial_number = _device.get_serialnum();
  pc::logger->info("k4a {} Open", device_index);

  _k4a_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
  _k4a_config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
  _k4a_config.color_resolution = K4A_COLOR_RESOLUTION_720P;
  _k4a_config.synchronized_images_only = true;

  _k4a_config.depth_mode = config.k4a.depth_mode;

  if (config.k4a.depth_mode == K4A_DEPTH_MODE_WFOV_UNBINNED) {
    _k4a_config.camera_fps = K4A_FRAMES_PER_SECOND_15;
  } else {
    _k4a_config.camera_fps = K4A_FRAMES_PER_SECOND_30;
  }

  _device.start_cameras(&_k4a_config);
  pc::logger->info("k4a {} started camera", device_index);

  _device.set_color_control(K4A_COLOR_CONTROL_EXPOSURE_TIME_ABSOLUTE,
                            K4A_COLOR_CONTROL_MODE_AUTO, 100000);

  _device.start_imu();
  pc::logger->info("k4a {} started IMU", device_index);

  // need to store the calibration and transformation data, as they are used
  // later to transform colour camera to point cloud space
  _calibration = _device.get_calibration(_k4a_config.depth_mode,
                                         _k4a_config.color_resolution);
  _transformation = k4a::transformation(_calibration);

  // start a thread that captures new frames and dumps them into raw buffers
  _capture_loop = std::thread(&K4ADriver::capture_frames, this);

  // _tracker = k4abt::tracker::create(_calibration);
  // _tracker_loop = std::thread(&K4ADriver::track_bodies, this);
  pc::logger->error("No Tracker!");

  _open = true;

  device_count++;
}

K4ADriver::~K4ADriver() {
  pc::logger->info("Closing driver for k4a {} ({})", device_index, id());
  _open = false;
  _stop_requested = true;
  _capture_loop.join();
  // _tracker_loop.join();
  _device.stop_cameras();
  _device.stop_imu();
  _device.close();
  device_count--;
}

void K4ADriver::reload() {
  _stop_requested = true;
  _capture_loop.join();
  // _tracker_loop.join();
  _stop_requested = false;

  _device.stop_cameras();
  _device.stop_imu();

  _tracker.destroy();

  _point_cloud = {};
  _skeletons.clear();

  _device.start_cameras(&_k4a_config);
  _device.start_imu();
  _open = true;

  _calibration = _device.get_calibration(_k4a_config.depth_mode,
                                         _k4a_config.color_resolution);
  _transformation = k4a::transformation(_calibration);
  _tracker = k4abt::tracker::create(_calibration);

  _capture_loop = std::thread(&K4ADriver::capture_frames, this);
  // _tracker_loop = std::thread(&K4ADriver::track_bodies, this);
}

bool K4ADriver::is_open() const { return _open; }

std::string K4ADriver::id() const { return _serial_number; }

void K4ADriver::set_paused(bool paused) { _pause_sensor = paused; }

void K4ADriver::enable_body_tracking(const bool enabled) {
  if (_body_tracking_enabled != enabled)
    _skeletons.clear();
  _body_tracking_enabled = enabled;
}

void K4ADriver::capture_frames() {
  while (!_stop_requested) {

    if (!_open || _pause_sensor) {
      std::this_thread::sleep_for(10ms);
      continue;
    }

    k4a::capture capture;
    try {
      bool result = _device.get_capture(&capture, 10ms);
      if (!result)
        continue;
    } catch (k4a::error e) {
      pc::logger->error(e.what());
      continue;
    }

    if (!_open)
      continue;

    k4a::image transformed_color_image;
    k4a::image point_cloud_image;

    try {
      if (is_aligning())
        run_aligner(capture);
      // if (_body_tracking_enabled) _tracker.enqueue_capture(capture);

      auto depth_image = capture.get_depth_image();
      auto color_image = capture.get_color_image();

      transformed_color_image =
          _transformation.color_image_to_depth_camera(depth_image, color_image);
      point_cloud_image = _transformation.depth_image_to_point_cloud(
          depth_image, K4A_CALIBRATION_TYPE_DEPTH);

    } catch (k4a::error e) {
      pc::logger->error(e.what());
      continue;
    }

    std::lock_guard<std::mutex> lock(_buffer_mutex);

    std::memcpy(_colors_buffer.data(), transformed_color_image.get_buffer(),
                color_buffer_size);
    std::memcpy(_positions_buffer.data(), point_cloud_image.get_buffer(),
                positions_buffer_size);

    _buffers_updated = true;
  }
}

void K4ADriver::track_bodies() {

  using namespace Eigen;

  // reserve space for five skeletons
  _skeletons.reserve(5);

  while (!_stop_requested) {

    k4abt_frame_t body_frame_handle = nullptr;
    k4abt::frame body_frame(body_frame_handle);
    if (!_tracker.pop_result(&body_frame, 50ms))
      continue;

    const auto body_count = body_frame.get_num_bodies();
    if (body_count == 0)
      continue;

    auto config = _last_config;
    // TODO make all the following serialize into the config
    auto auto_tilt = _auto_tilt;
    auto alignment_center = Eigen::Vector3f(
        _alignment_center.x, _alignment_center.y, _alignment_center.z);
    auto aligned_position_offset =
        Eigen::Vector3f(_aligned_position_offset.x, _aligned_position_offset.y,
                        _aligned_position_offset.z);

    // transform the skeletons based on device config
    // and place them into the _skeletons list
    _skeletons.clear();
    for (std::size_t body_num = 0; body_num < body_count; body_num++) {
      const k4abt_skeleton_t raw_skeleton = body_frame.get_body_skeleton(0);
      K4ASkeleton skeleton;
      // parse each joint
      for (std::size_t joint = 0; joint < K4ABT_JOINT_COUNT; joint++) {
        auto pos = raw_skeleton.joints[joint].position.xyz;
        auto orientation = raw_skeleton.joints[joint].orientation.wxyz;

        Vector3f pos_f(pos.x, pos.y, pos.z);
        Quaternionf ori_f(orientation.w, orientation.x, orientation.y,
                          orientation.z);

        // perform any auto-tilt
        pos_f = auto_tilt * pos_f;
        ori_f = auto_tilt * ori_f;
        // flip y and z axes for our world space
        pos_f = Vector3f(pos_f[0], -pos_f[1], -pos_f[2]);

        static constexpr auto rad = [](float deg) {
          constexpr auto mult = 3.141592654f / 180.0f;
          return deg * mult;
        };

        // create the rotation around our center
        AngleAxisf rot_x(rad(config.rotation_deg.x), Vector3f::UnitX());
        AngleAxisf rot_y(rad(config.rotation_deg.y), Vector3f::UnitY());
        AngleAxisf rot_z(rad(config.rotation_deg.z), Vector3f::UnitZ());
        Quaternionf q = rot_z * rot_y * rot_x;
        Affine3f rot_transform = Translation3f(-alignment_center) * q *
                                 Translation3f(alignment_center);

        // perform manual rotation
        pos_f = rot_transform * pos_f;
        ori_f = q * ori_f;

        // then alignment translation
        pos_f += alignment_center + aligned_position_offset;

        // perform our manual translation
        pos_f += Vector3f(config.offset.x, config.offset.y, config.offset.z);

        // specified axis flips
        if (config.flip_x) {
          pos_f.x() = -pos_f.x();
          ori_f *= Quaternionf(AngleAxisf(M_PI, Vector3f::UnitX()));
        }
        if (config.flip_y) {
          pos_f.y() = -pos_f.y();
          ori_f *= Quaternionf(AngleAxisf(M_PI, Vector3f::UnitY()));
        }
        if (config.flip_z) {
          pos_f.z() = -pos_f.z();
          ori_f *= Quaternionf(AngleAxisf(M_PI, Vector3f::UnitZ()));
        }

        // and scaling
        pos_f *= config.scale;

        position pos_out = {static_cast<short>(std::round(pos_f.x())),
                            static_cast<short>(std::round(pos_f.y())),
                            static_cast<short>(std::round(pos_f.z())), 0};

        skeleton[joint].first = {pos_out.x, pos_out.y, pos_out.z};
        skeleton[joint].second = {ori_f.w(), ori_f.x(), ori_f.y(), ori_f.z()};
      }
      _skeletons.push_back(skeleton);
    }
  }
}

void K4ADriver::apply_auto_tilt(const bool apply) {
  if (!apply) {
    _auto_tilt = Eigen::Matrix3f::Identity();
    return;
  }

  static std::future<void> auto_tilt_task;

  if (auto_tilt_task.valid() &&
      auto_tilt_task.wait_for(0ms) != std::future_status::ready) {
    // compute is still running
    return;
  }

  auto_tilt_task = std::async(std::launch::async, [&] {
    constexpr auto sample_count = 50;
    std::vector<std::array<float, 3>> accelerometer_samples;
    for (int i = 0; i < sample_count; i++) {
      if (i != 0)
        std::this_thread::sleep_for(100us);
      accelerometer_samples.push_back(accelerometer_sample());
    }

    // and get the average
    std::array<float, 3> sum = std::accumulate(
        accelerometer_samples.begin(), accelerometer_samples.end(),
        std::array<float, 3>{0.0f, 0.0f, 0.0f},
        [](const auto &a, const auto &b) {
          return std::array<float, 3>{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
        });
    std::array<float, 3> accel_average{
        sum[0] / sample_count, sum[1] / sample_count, sum[2] / sample_count};

    // and turn it into a rotation matrix we can apply
    Eigen::Quaternionf q;
    q.setFromTwoVectors(
        Eigen::Vector3f(accel_average[1], accel_average[2], accel_average[0]),
        Eigen::Vector3f(0.f, -1.f, 0.f));

    Eigen::Matrix3f last_tilt = _auto_tilt;
    Eigen::Matrix3f new_tilt = q.toRotationMatrix().transpose();

    float tilt_difference =
        (new_tilt * last_tilt.transpose()).eulerAngles(2, 1, 0).norm();

    constexpr float difference_threshold = M_PI / 180.0f * 0.25f; // degrees

    if (tilt_difference > difference_threshold) {
      _auto_tilt = new_tilt;
    }
  });
}

void K4ADriver::start_alignment() {
  pc::logger->info("Beginning alignment for k4a {} ({})", device_index, id());
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

void K4ADriver::set_depth_mode(const k4a_depth_mode_t mode) {
  _open = false;
  _k4a_config.depth_mode = mode;
  if (mode == K4A_DEPTH_MODE_WFOV_UNBINNED) {
    _k4a_config.camera_fps = K4A_FRAMES_PER_SECOND_15;
  } else {
    _k4a_config.camera_fps = K4A_FRAMES_PER_SECOND_30;
  }
  reload();
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

std::array<float, 3> K4ADriver::accelerometer_sample() {
  k4a_imu_sample_t imu_sample;

  if (!_device.get_imu_sample(&imu_sample, 100ms))
    return {0, 0, 0};

  auto accel = imu_sample.acc_sample.v;
  return {accel[0], accel[1], accel[2]};
}

} // namespace pc::devices
