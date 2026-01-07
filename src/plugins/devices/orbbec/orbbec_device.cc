#include "orbbec_device.h"
#include "orbbec_context.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/h/ObTypes.h>
#include <libobsensor/hpp/Context.hpp>
#include <libobsensor/hpp/Error.hpp>
#include <libobsensor/hpp/Utils.hpp>
#include <memory>
#include <print>
#include <thread>

using namespace std::chrono;
using namespace std::chrono_literals;

namespace pc::devices {

OrbbecDevice::OrbbecDevice(Corrade::PluginManager::AbstractManager &manager,
                           Corrade::Containers::StringView plugin)
    : DevicePlugin(manager, plugin) {
  try {
    orbbec_context().retain_user();
    orbbec_context().discover_devices_async();
  } catch (...) {
    std::println("Exception during Orbbec context initialisation");
  }

  // do our device initialisation / start procedure when
  // the context is known to be ready
  orbbec_context().run_on_ready([this] {
    notify_status_changed();
    start_sync();
    _timeout_thread = std::jthread(&OrbbecDevice::timeout_thread_work, this);
  });

  std::println("Created OrbbecDevice");
}

OrbbecDevice::~OrbbecDevice() {
  stop_sync();
  _timeout_thread.request_stop();
  orbbec_context().release_user();
  std::println("Destroyed OrbbecDevice");
}

DeviceStatus OrbbecDevice::status() const {
  auto ctx = orbbec_context().get_if_ready();
  if (!ctx) return DeviceStatus::Unloaded;
  if (_in_error_state) return DeviceStatus::Missing;
  if (!_running_pipeline) return DeviceStatus::Loaded;
  if (_last_updated_time.load() == std::chrono::steady_clock::time_point{}) {
    return DeviceStatus::Loaded;
  }
  return DeviceStatus::Active;
}

void OrbbecDevice::start() {
  _initialisation_thread = std::jthread([this](std::stop_token stop_token) {
    // TODO stop_token is currently unused,
    // should pass it in and check at different initialisation thread steps
    start_sync();
  });
}

void OrbbecDevice::stop() {
  _initialisation_thread = std::jthread([this](std::stop_token stop_token) {
    // TODO stop_token is currently unused,
    // should pass it in and check at different initialisation thread steps
    stop_sync();
  });
}

void OrbbecDevice::restart() {
  _initialisation_thread = std::jthread([this](std::stop_token stop_token) {
    // TODO stop_token is currently unused,
    // should pass it in and check at different initialisation thread steps
    stop_sync();
    start_sync();
  });
}

void OrbbecDevice::restart_sync() {
  stop_sync();
  start_sync();
}

void OrbbecDevice::start_sync() {
  std::println("attempting to start an orrbec driver");

  std::lock_guard lock(orbbec_context().start_stop_device_access);

  set_running(false);
  set_error_state(false);
  set_updated_time(steady_clock::time_point{});

  auto &config = std::get<OrbbecDeviceConfiguration>(this->config());
  // pc::logger->info("Initialising OrbbecDevice at {}", config.ip);
  std::println("Initialising OrbbecDevice at {}", config.ip);

  constexpr uint16_t net_device_port = 8090; // femto mega default

  bool existing_device = false;

  auto ob_ctx = orbbec_context().wait_til_ready();
  if (!ob_ctx) {
    std::println("Orbbec context not ready in time, aborting start_sync()");
    return;
  }

  auto ob_device_list = ob_ctx->queryDeviceList();
  auto count = ob_device_list->deviceCount();
  for (size_t i = 0; i < count; i++) {
    auto device = ob_device_list->getDevice(i);
    auto info = device->getDeviceInfo();
    if (strcmp(info->ipAddress(), config.ip.c_str()) == 0) {
      _ob_device = device;
      existing_device = true;
    }
  }

  if (!existing_device) {
    try {
      _ob_device = ob_ctx->createNetDevice(config.ip.data(), net_device_port);
    } catch (const ob::Error &e) {
      // pc::logger->error("Failed to create OrbbecDevice at {}:{}", config.ip,
      //                   net_device_port);
      std::println("Failed to create OrbbecDevice at {}:{}", config.ip,
                   net_device_port);
      return;
    } catch (...) {
      // pc::logger->error("Unknown error creating Orrbec NetDevice");
      std::println("Unknown error creating Orrbec NetDevice");
    }
  }

  _ob_pipeline = std::make_shared<ob::Pipeline>(_ob_device);
  _ob_config = std::make_shared<ob::Config>();
  _ob_point_cloud = std::make_unique<ob::PointCloudFilter>();

  _ob_xy_tables.reset();
  _xy_table_data.reset();
  _depth_positions_buffer.reset();
  _depth_frame_buffer.reset();

  constexpr auto fps = 30;
  constexpr auto color_width = 1280;
  constexpr auto color_height = 720;

  // Default to using Narrow FOV
  auto depth_width = 640;
  auto depth_height = 576;
  // Wide FOV depth mode
  if (config.depth_mode == 1) {
    depth_width = 512;
    depth_height = 512;
  }

  // default acquisition mode is RGBD
  if (config.acquisition_mode == 0) {

    _ob_config->setAlignMode(ALIGN_D2C_HW_MODE);
    _ob_config->setFrameAggregateOutputMode(
        OB_FRAME_AGGREGATE_OUTPUT_FULL_FRAME_REQUIRE);

    auto color_profiles = _ob_pipeline->getStreamProfileList(OB_SENSOR_COLOR);
    auto color_profile = color_profiles->getVideoStreamProfile(
        color_width, color_height, OB_FORMAT_MJPG, fps);

    _ob_config->enableStream(color_profile);

    auto depth_profiles =
        _ob_pipeline->getD2CDepthProfileList(color_profile, ALIGN_D2C_HW_MODE);
    auto depth_profile = depth_profiles->getVideoStreamProfile(
        depth_width, depth_height, OB_FORMAT_Y16, fps);

    _ob_config->enableStream(depth_profile);

    if (!init_device_memory(color_profile->width() * color_profile->height())) {
      // pc::logger->error("Failed to initialise GPU memory for Orbbec
      // pipeline");
      std::println("Failed to initialise GPU memory for Orbbec pipeline");
      return;
    };

    std::println("about to start pipeline");

    _ob_pipeline->start(
        _ob_config, [&](std::shared_ptr<ob::FrameSet> frame_set) {
          auto color_frame = frame_set->colorFrame();
          auto point_count = color_frame->width() * color_frame->height();
          std::shared_ptr<ob::Frame> point_cloud;
          try {
            point_cloud = _ob_point_cloud->process(frame_set);
          } catch (ob::Error e) {
            std::println("Failed to process Orbbec frame: {}", e.what());
            return;
          }
          if (point_cloud && point_cloud->data()) {
            std::lock_guard lock(_point_buffer_access);
            _point_buffer.resize(point_count);
            std::memcpy(_point_buffer.data(), point_cloud->data(),
                        point_count * sizeof(OBColorPoint));
          }
          set_updated_time(steady_clock::now());
          _buffer_updated = true;
        });

    _ob_point_cloud->setCameraParam(_ob_pipeline->getCameraParam());
    _ob_point_cloud->setCreatePointFormat(OB_FORMAT_RGB_POINT);

    set_error_state(false);
    set_running(true);
  }

  // depth_only acquisition_mode
  else if (config.acquisition_mode == 1) {

    auto depth_profiles = _ob_pipeline->getStreamProfileList(OB_SENSOR_DEPTH);
    auto depth_profile = depth_profiles->getVideoStreamProfile(
        depth_width, depth_height, OB_FORMAT_Y16, fps);

    _ob_config->enableStream(depth_profile);
    _ob_config->setAlignMode(ALIGN_DISABLE);

    if (!init_device_memory(depth_width * depth_height)) {
      std::println("Failed to initialise GPU memory for Orbbec pipeline");
      // pc::logger->error("Failed to initialise GPU memory for Orbbec
      // pipeline");
      return;
    }

    _point_buffer.resize(depth_width * depth_height);

    auto camera_param = _ob_pipeline->getCameraParam();
    _ob_point_cloud->setCameraParam(camera_param);

    auto calibration_param = _ob_pipeline->getCalibrationParam(_ob_config);
    auto stream_profile = depth_profile->as<ob::VideoStreamProfile>();

    uint32_t xy_table_size = depth_width * depth_height * 2;
    _xy_table_data = std::vector<float>(xy_table_size);
    _ob_xy_tables = OBXYTables{};
    bool table_init = ob::CoordinateTransformHelper::transformationInitXYTables(
        calibration_param, OB_SENSOR_DEPTH, (*_xy_table_data).data(),
        &xy_table_size, &(*_ob_xy_tables));

    if (!table_init) {
      std::println("Failed to initialise orbbec transformation tables");
      // pc::logger->error("Failed to initialise orbbec transformation tables");
      return;
    }

    const auto frame_size = depth_height * depth_width;
    _depth_positions_buffer = std::vector<OBPoint3f>(frame_size);
    _depth_frame_buffer = std::vector<OBColorPoint>(frame_size);

    _ob_pipeline->start(
        _ob_config,
        [this, frame_size](std::shared_ptr<ob::FrameSet> frame_set) {
          if (frame_set == nullptr || frame_set->depthFrame() == nullptr)
            return;

          auto depth_frame = frame_set->depthFrame();
          auto *depth_data = static_cast<uint16_t *>(depth_frame->data());

          auto &depth_positions_buffer = (*_depth_positions_buffer);
          auto &depth_frame_buffer = (*_depth_frame_buffer);

          ob::CoordinateTransformHelper::transformationDepthToPointCloud(
              &(*_ob_xy_tables), depth_data, depth_positions_buffer.data());

          std::println("this acquisition mode isnt implemented");

          // tbb::parallel_for(
          //     tbb::blocked_range<size_t>(0, frame_size), [&](auto &r) {
          //       for (size_t i = r.begin(); i < r.end(); ++i) {
          //         auto pos = depth_positions_buffer[i];
          //         depth_frame_buffer[i] =
          //             OBColorPoint{pos.x, pos.y, pos.z, 255, 255, 255};
          //       }
          //     });

          {
            std::lock_guard lock(_point_buffer_access);
            std::swap(_point_buffer, depth_frame_buffer);
            _buffer_updated = true;
            set_updated_time(steady_clock::now());
          }
        });

    set_error_state(false);
    set_running(true);
  }
}

void OrbbecDevice::stop_sync() {
  auto &config = std::get<OrbbecDeviceConfiguration>(this->config());
  std::lock_guard lock(orbbec_context().start_stop_device_access);
  std::println("Closing OrbbecDevice {}", config.ip);
  // pc::logger->info("Closing OrbbecDevice {}", config().ip);
  set_running(false);
  set_updated_time(steady_clock::time_point{});
  if (_ob_pipeline) _ob_pipeline->stop();
  if (_device_memory_ready.load()) free_device_memory();
}

void OrbbecDevice::timeout_thread_work(std::stop_token stop_token) {
  constexpr auto error_timeout = 10s;
  constexpr auto recovery_window = 500ms;
  constexpr auto check_interval = 1s;

  while (!stop_token.stop_requested()) {

    const auto now = steady_clock::now();
    const auto last_frame_time =
        _last_updated_time.load(std::memory_order_relaxed);

    const bool has_seen_a_frame = last_frame_time != steady_clock::time_point{};

    if (_running_pipeline && !_in_error_state) {
      if (!has_seen_a_frame) {
        std::this_thread::sleep_for(check_interval);
        continue;
      }
      if (now - last_frame_time >= error_timeout) {
        set_error_state(true);
        auto &config = std::get<OrbbecDeviceConfiguration>(this->config());
        std::println(
            "Orbbec device '{}' entered error state. Attempting restart...",
            config.id);
        restart();
      }
    } else if (_in_error_state) {
      if (has_seen_a_frame && (now - last_frame_time <= recovery_window)) {
        set_error_state(false);
      }
    }
    std::this_thread::sleep_for(check_interval);
  }
}

// TODO in cuda
bool OrbbecDevice::init_device_memory(std::size_t incoming_point_count) {
  return true;
}

void OrbbecDevice::free_device_memory() {}

} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(OrbbecDevice, pc::devices::OrbbecDevice,
                        "net.pointcaster.DevicePlugin/1.0")
