#include "orbbec_device.h"
#include "orbbec_context.h"
#include "orbbec_device_config.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/h/ObTypes.h>
#include <libobsensor/hpp/Context.hpp>
#include <libobsensor/hpp/Error.hpp>
#include <libobsensor/hpp/Pipeline.hpp>
#include <libobsensor/hpp/Utils.hpp>
#include <memory>
#include <print>
#include <thread>

#include <metrics/metrics.h>

using namespace std::chrono;
using namespace std::chrono_literals;

namespace {
constexpr int colour_width = 1280;
constexpr int colour_height = 720;
constexpr int depth_width = 640;
constexpr int depth_height = 576;
constexpr int fps = 30;
} // namespace

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
    _timeout_thread = std::jthread([this](auto stop_token) {
      // &OrbbecDevice::timeout_thread_work
      timeout_thread_work(stop_token);
    });
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
  if (_loading_pipeline) return DeviceStatus::Loading;
  auto ctx = orbbec_context().get_if_ready();
  if (!ctx) return DeviceStatus::Unloaded;
  if (_in_error_state) return DeviceStatus::Missing;
  if (!_running_pipeline) {
    std::println("returning loaded from pipeline");
    return DeviceStatus::Loaded;
  }
  if (_last_updated_time.load() == std::chrono::steady_clock::time_point{}) {
    std::println("returning loaded from clock");
    return DeviceStatus::Loaded;
  }
  std::println("returning active");
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

  set_loading(true);
  set_running(false);
  set_error_state(false);
  set_updated_time(steady_clock::time_point{});

  const auto &config = std::get<OrbbecDeviceConfiguration>(this->config());
  std::println("Initialising OrbbecDevice at {}", config.ip);

  auto ob_ctx = orbbec_context().wait_til_ready();
  if (!ob_ctx) {
    std::println("Orbbec context not ready in time, aborting start_sync()");
    set_loading(false);
    set_error_state(true);
    return;
  }

  std::shared_ptr<ob::Device> ob_device;

  // TODO maybe try via mac address before trying by ip?

  // try to find existing device in the list by network IP
  if (auto ob_device_list = ob_ctx->queryDeviceList()) {
    const size_t device_count = ob_device_list->deviceCount();
    for (size_t i = 0; i < device_count; ++i) {
      auto device = ob_device_list->getDevice(i);
      if (!device) continue;

      auto info = device->getDeviceInfo();
      if (!info) continue;

      if (!config.ip.empty() && info->ipAddress()) {
        if (std::strcmp(info->ipAddress(), config.ip.c_str()) == 0) {
          ob_device = device;
          break;
        }
      }
    }
  }

  if (!ob_device && !config.ip.empty()) {
    constexpr uint16_t net_device_port = 8090; // Femto Mega default
    try {
      ob_device = ob_ctx->createNetDevice(config.ip.c_str(), net_device_port);
    } catch (const ob::Error &e) {
      std::println("Failed to create OrbbecDevice at {}:{}: {}", config.ip,
                   net_device_port, e.what());
      set_error_state(true);
      return;
    } catch (...) {
      std::println("Unknown error creating Orbbec NetDevice at {}:{}",
                   config.ip, net_device_port);
      set_error_state(true);
      return;
    }
  }

  if (!ob_device) {
    std::println("Warning: Unable to find Orbbec at '{}'", config.ip);
    set_error_state(true);
    return;
  }

  if (config.acquisition_mode ==
      OrbbecDeviceConfiguration::AcquisitionMode::XYZRGB) {
    if (!init_device_memory(colour_width * colour_height)) {
      std::println("Failed to initialise GPU memory for Orbbec pipeline");
      set_error_state(true);
      return;
    }
  }
  //
  else if (config.acquisition_mode ==
           OrbbecDeviceConfiguration::AcquisitionMode::XYZ) {
    std::println("acquisition_mode not implemented yet");
    set_error_state(true);
    return;
  }

  std::println("about to start pipeline");
  _pipeline_thread =
      std::jthread([this, ob_device = std::move(ob_device)](auto stop_token) {
        pipeline_thread_work(stop_token, ob_device);
      });

  set_running(true);
  set_error_state(false);
}

void OrbbecDevice::stop_sync() {
  auto &config = std::get<OrbbecDeviceConfiguration>(this->config());
  std::lock_guard lock(orbbec_context().start_stop_device_access);
  std::println("Closing OrbbecDevice {}", config.ip);
  // pc::logger->info("Closing OrbbecDevice {}", config().ip);
  _pipeline_thread.request_stop();
  _pipeline_thread.join();
  if (_device_memory_ready.load()) free_device_memory();
  set_running(false);
  set_updated_time(steady_clock::time_point{});
}

void OrbbecDevice::set_updated_time(
    std::chrono::steady_clock::time_point new_time) {
  const auto old_time = _last_updated_time.exchange(new_time);

  const bool new_invalid =
      (new_time == std::chrono::steady_clock::time_point{});
  const bool old_invalid =
      (old_time == std::chrono::steady_clock::time_point{});

  if (!new_invalid) {
    const auto prev_tick = _pipeline_last_tick.exchange(new_time);

    if (prev_tick != std::chrono::steady_clock::time_point{}) {
      const std::chrono::duration<float> dt = new_time - prev_tick;
      if (dt.count() > 0.0f) {
        static constexpr float pipeline_fps_ema_alpha = 0.10f;

        const float inst_fps = 1.0f / dt.count();
        const float old_ema = _pipeline_fps_ema.load();
        const float new_ema = (old_ema == 0.0f)
                                  ? inst_fps
                                  : (pipeline_fps_ema_alpha * inst_fps +
                                     (1.0f - pipeline_fps_ema_alpha) * old_ema);
        _pipeline_fps_ema.store(new_ema);
      }
    }
  } else {
    _pipeline_last_tick.store(std::chrono::steady_clock::time_point{});
    _pipeline_fps_ema.store(0.0f);
  }

  if (new_invalid != old_invalid) notify_status_changed();
}

void OrbbecDevice::pipeline_thread_work(std::stop_token stop_token,
                                        std::shared_ptr<ob::Device> ob_device) {
  try {
    ob::Pipeline pipeline{ob_device};

    auto ob_config = std::make_shared<ob::Config>();

    auto colour_profile_list = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
    // TODO enable without colour too
    if (!colour_profile_list) {
      std::println("No Orbbec colour profiles available");
      set_error_state(true);
      return;
    }
    std::shared_ptr<ob::VideoStreamProfile> colour_profile;
    try {
      colour_profile = colour_profile_list->getVideoStreamProfile(
          colour_width, colour_height, OB_FORMAT_MJPG, fps);
    } catch (const ob::Error &e) {
      std::println("Failed to get colour profile: {}", e.what());
      set_error_state(true);
      return;
    }
    ob_config->enableStream(colour_profile);

    std::shared_ptr<ob::VideoStreamProfile> depth_profile;

    // try hardware D2C first
    std::shared_ptr<ob::StreamProfileList> depth_d2c_list;
    try {
      depth_d2c_list =
          pipeline.getD2CDepthProfileList(colour_profile, ALIGN_D2C_HW_MODE);
    } catch (const ob::Error &e) {
      std::println("Failed to get HW D2C depth profile list: {}", e.what());
    }

    if (depth_d2c_list && depth_d2c_list->count() > 0) {
      try {
        depth_profile = depth_d2c_list->getVideoStreamProfile(
            depth_width, depth_height, OB_FORMAT_Y16, fps);
        ob_config->setAlignMode(ALIGN_D2C_HW_MODE);
      } catch (const ob::Error &e) {
        std::println("Failed to select HW D2C depth profile: {}", e.what());
      }
    }

    // fallback to software D2C if HW failed or unsupported for this combo
    if (!depth_profile) {
      std::println("Falling back to ALIGN_D2C_SW_MODE");
      try {
        auto depth_sw_list =
            pipeline.getD2CDepthProfileList(colour_profile, ALIGN_D2C_SW_MODE);
        if (!depth_sw_list || depth_sw_list->count() == 0) {
          std::println("No SW D2C depth profiles available");
          set_error_state(true);
          return;
        }
        depth_profile = depth_sw_list->getVideoStreamProfile(
            depth_width, depth_height, OB_FORMAT_Y16, fps);
        ob_config->setAlignMode(ALIGN_D2C_SW_MODE);
      } catch (const ob::Error &e) {
        std::println("Failed to select SW D2C depth profile: {}", e.what());
        set_error_state(true);
        return;
      }
    }

    ob_config->enableStream(depth_profile);
    ob_config->setFrameAggregateOutputMode(
        OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);

    pipeline.enableFrameSync();

    try {
      pipeline.start(ob_config);
    } catch (const ob::Error &e) {
      std::println("Failed to start Orbbec pipeline: {}", e.what());
      set_error_state(true);
      return;
    }

    ob::PointCloudFilter point_cloud_filter;
    point_cloud_filter.setCameraParam(pipeline.getCameraParam());

    // TODO: query actual depth unit scale from device
    constexpr float depth_position_scale = 1.0f;
    point_cloud_filter.setPositionDataScaled(depth_position_scale);

    point_cloud_filter.setCreatePointFormat(OB_FORMAT_RGB_POINT);

    // processing loop
    while (!stop_token.stop_requested()) {
      std::shared_ptr<ob::FrameSet> frame_set;
      try {
        frame_set = pipeline.waitForFrameset(100);
      } catch (const ob::Error &e) {
        std::println("waitForFrameset error: {}", e.what());
        continue;
      }

      if (!frame_set) continue;
      if (_loading_pipeline.load(std::memory_order_relaxed)) set_loading(false);

      auto colour_frame = frame_set->colorFrame();
      auto depth_frame = frame_set->depthFrame();
      if (!colour_frame || !depth_frame) continue;

      const uint32_t width = colour_frame->width();
      const uint32_t height = colour_frame->height();
      const size_t point_count = static_cast<size_t>(width) * height;

      std::shared_ptr<ob::Frame> point_cloud_frame;
      try {
        point_cloud_frame = point_cloud_filter.process(frame_set);
      } catch (const ob::Error &e) {
        std::println("Failed to process Orbbec frame: {}", e.what());
        continue;
      } catch (const std::exception &e) {
        std::println("Failed to process Orbbec frame: {}", e.what());
        continue;
      } catch (...) {
        std::println("Unknown exception in Orbbec process()");
        continue;
      }

      if (point_cloud_frame && point_cloud_frame->data()) {
        std::lock_guard buffer_lock(_point_buffer_access);
        _point_buffer.resize(point_count);
        std::memcpy(_point_buffer.data(), point_cloud_frame->data(),
                    point_count * sizeof(OBColorPoint));
        _buffer_updated = true;
        set_updated_time(steady_clock::now());
      }
    }
  } catch (const std::exception &e) {
    std::println("Exception in Orbbec processing thread: {}", e.what());
    set_error_state(true);
  } catch (...) {
    std::println("Unknown exception in Orbbec processing thread");
    set_error_state(true);
  }
}

void OrbbecDevice::timeout_thread_work(std::stop_token stop_token) {
  constexpr auto error_timeout = 10s;
  constexpr auto recovery_window = 500ms;
  constexpr auto check_interval = 1s;

  auto &config = std::get<OrbbecDeviceConfiguration>(this->config());

  while (!stop_token.stop_requested()) {

    const auto now = steady_clock::now();
    const auto last_frame_time =
        _last_updated_time.load(std::memory_order_relaxed);

    const bool has_seen_a_frame = last_frame_time != steady_clock::time_point{};

    // we can update metrics on this thread too
    pc::metrics::set_gauge("pointcaster_orbbec_pipeline_hertz",
                           _pipeline_fps_ema.load(),
                           {{"device_id", config.id}});

    if (_running_pipeline && !_in_error_state) {
      if (!has_seen_a_frame) {
        std::this_thread::sleep_for(check_interval);
        continue;
      }
      if (now - last_frame_time >= error_timeout) {
        set_error_state(true);
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
