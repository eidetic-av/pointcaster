#include "orbbec_device.h"
#include "orbbec_context.h"
#include "orbbec_device_config.h"

#include <Corrade/Containers/Pointer.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <core/logger/logger.h>
#include <core/profiling/profiling_zone.h>
#include <cstdint>
#include <cstring>
#include <execution>
#include <libobsensor/ObSensor.hpp>
#include <libobsensor/h/ObTypes.h>
#include <libobsensor/hpp/Context.hpp>
#include <libobsensor/hpp/Error.hpp>
#include <libobsensor/hpp/Pipeline.hpp>
#include <libobsensor/hpp/Utils.hpp>
#include <memory>
#include <metrics/metrics.h>
#include <mutex>
#include <numeric>
#include <plugins/devices/device_variants.h>
#include <pointcaster/point_cloud.h>
#include <profiling/profiler.h>
#include <random>
#include <span>
#include <thread>
#include <util/string_utils.h>
#include <variant>
#include <workspace/workspace.h>

using namespace std::chrono;
using namespace std::chrono_literals;

using pc::profiling::ProfilingZone;

namespace {
std::atomic_bool does_have_discovery_change_callback = false;
} // namespace

namespace pc::devices {

OrbbecDevice::OrbbecDevice(Corrade::PluginManager::AbstractManager &manager,
                           Corrade::Containers::StringView plugin)
    : DevicePlugin(manager, plugin) {
  pc::logger()->trace("Creating new OrbbecDevice");
  try {
    orbbec_context().retain_user();
    orbbec_context().discover_devices_async();
  } catch (...) {
    pc::logger()->error("Exception during Orbbec context initialisation");
  }
  _timeout_thread = std::jthread([this](auto stop_token) {
    // &OrbbecDevice::timeout_thread_work
    timeout_thread_work(stop_token);
  });
  pc::logger()->trace("Created OrbbecDevice");
}

OrbbecDevice::~OrbbecDevice() {
  pc::logger()->debug("~OrbbecDevice Destructor");
  stop_sync();
  _timeout_thread.request_stop();
  orbbec_context().release_user();
  pc::logger()->trace("Destroyed OrbbecDevice{}",
                      _is_discovery_instance ? " discovery instance" : "");
}

void OrbbecDevice::init(Workspace &workspace) {
  _workspace = &workspace;
  // do our device initialisation / start procedure when
  // the context is known to be ready
  orbbec_context().run_on_ready([this] {
    notify_status_changed();
    start();
  });
}

std::vector<DiscoveredDevice> OrbbecDevice::discovered_devices() const {
  std::vector<DiscoveredDevice> out;

  auto &ctx = orbbec_context();
  std::lock_guard lock(ctx.discovered_devices_access);

  out.reserve(ctx.discovered_devices.size());
  for (const auto &d : ctx.discovered_devices) {
    out.push_back(DiscoveredDevice{
        .label = std::format("{} ({})", d.name, d.ip), .ip = d.ip, .id = d.id});
  }

  return out;
}

void OrbbecDevice::refresh_discovery() {
  orbbec_context().discover_devices_async();
}

void OrbbecDevice::add_discovery_change_callback(std::function<void()> cb) {
  orbbec_context().add_discovery_change_callback(std::move(cb));
  does_have_discovery_change_callback = true;
}

bool OrbbecDevice::has_discovery_change_callback() const {
  return does_have_discovery_change_callback.load();
}

DeviceStatus OrbbecDevice::status() const {
  if (_loading_pipeline) return DeviceStatus::Loading;
  auto ctx = orbbec_context().get_if_ready();
  if (!ctx) return DeviceStatus::Unloaded;
  if (_in_error_state) return DeviceStatus::Missing;
  if (!_running_pipeline) {
    return DeviceStatus::Loaded;
  }
  if (_last_updated_time.load() == std::chrono::steady_clock::time_point{}) {
    return DeviceStatus::Loaded;
  }
  return DeviceStatus::Active;
}

void OrbbecDevice::start() {
  pc::logger()->debug("Running 'start()'");
  _initialisation_thread = std::jthread([this](std::stop_token stop_token) {
    // TODO stop_token is currently unused,
    // should pass it in and check at different initialisation thread steps
    start_sync();
  });
}

void OrbbecDevice::stop() {
  pc::logger()->debug("Running 'stop()'");
  _initialisation_thread = std::jthread([this](std::stop_token stop_token) {
    // TODO stop_token is currently unused,
    // should pass it in and check at different initialisation thread steps
    stop_sync();
  });
}

void OrbbecDevice::restart() {
  pc::logger()->debug("Running 'restart()'");
  _initialisation_thread = std::jthread([this](std::stop_token stop_token) {
    // TODO stop_token is currently unused,
    // should pass it in and check at different initialisation thread steps
    stop_sync();
    start_sync();
  });
}

void OrbbecDevice::start_sync() {
  pc::logger()->debug("Running 'start_sync()'");
  pc::logger()->trace("Attempting to start an Orrbec driver");

  std::lock_guard lock(orbbec_context().start_stop_device_access);

  set_loading(true);
  set_running(false);
  set_error_state(false);
  set_updated_time(steady_clock::time_point{});

  auto ob_ctx = orbbec_context().wait_til_ready();
  if (!ob_ctx) {
    pc::logger()->trace(
        "Orbbec context not ready in time, aborting start_sync()");
    set_loading(false);
    set_error_state(true);
    return;
  }

  // if this plugin instance is only for context initialisation and device
  // discovery, dont bother trying to attach to a device
  if (is_discovery_instance()) return;

  const auto config = std::get<OrbbecDeviceConfiguration>(this->config());
  pc::logger()->info("Initialising OrbbecDevice ({})", config.id);

  std::shared_ptr<ob::Device> ob_device;

  // try to find existing device in the list by device UID
  if (auto ob_device_list = ob_ctx->queryDeviceList()) {
    const size_t device_count = ob_device_list->deviceCount();
    for (size_t i = 0; i < device_count; ++i) {
      auto id = ob_device_list->getUid(i);
      if (!config.id.empty() && id) {
        if (std::strcmp(id, config.id.c_str()) == 0) {
          try {
            ob_device = ob_device_list->getDevice(i);
            pc::logger()->trace("Completed ob_device_list->getDevice({})", i);
          } catch (...) {
            pc::logger()->warn("Unable to open device with matched device UID");
            pc::logger()->warn("Check connection to camera");
          }
          break;
        }
      }
    }
  }

  constexpr uint16_t net_device_port = 8090; // Femto Mega default
  const auto &ip = config.network.ip_address.value();

  if (!ob_device && !ip.empty()) {
    try {
      pc::logger()->trace("Attempting createNetDevice at {}:{}", ip,
                          net_device_port);
      ob_device = ob_ctx->createNetDevice(ip.c_str(), net_device_port);
    } catch (const ob::Error &e) {
      pc::logger()->error("Failed to create OrbbecDevice at {}:{}: {}",
                          config.network.ip_address.value(), net_device_port,
                          e.what());
      set_error_state(true);
      return;
    } catch (...) {
      pc::logger()->error("Unknown error creating Orbbec NetDevice at {}:{}",
                          config.network.ip_address.value(), net_device_port);
      set_error_state(true);
      return;
    }
  }

  if (!ob_device) {
    pc::logger()->warn("Unable to find Orbbec at {}:{}", ip, net_device_port);
    set_error_state(true);
    return;
  }

  pc::logger()->trace("Successfully created device at {}:{}", ip,
                      net_device_port);

  const auto [colour_width, colour_height] =
      orbbec::resolution(config.color_resolution);
  const auto [depth_width, depth_height] =
      orbbec::resolution(config.depth_resolution);

  if (config.acquisition_mode ==
      OrbbecDeviceConfiguration::AcquisitionMode::XYZRGB) {
    if (!init_device_memory(colour_width * colour_height)) {
      pc::logger()->error(
          "Failed to initialise GPU memory for Orbbec pipeline");
      set_error_state(true);
      return;
    }
  }
  //
  else if (config.acquisition_mode ==
           OrbbecDeviceConfiguration::AcquisitionMode::XYZ) {
    pc::logger()->warn("acquisition_mode not implemented yet");
    set_error_state(true);
    return;
  }

  _pipeline_thread =
      std::jthread([this, ob_device = std::move(ob_device)](auto stop_token) {
        pipeline_thread_work(stop_token, ob_device);
      });

  set_running(true);
  set_error_state(false);
}

void OrbbecDevice::stop_sync() {
  pc::logger()->debug("Running 'stop_sync()'");
  if (!_running_pipeline) return;
  auto config = std::get<OrbbecDeviceConfiguration>(this->config());
  std::lock_guard lock(orbbec_context().start_stop_device_access);
  pc::logger()->info("Closing OrbbecDevice {}",
                     config.network.ip_address.value());
  pc::logger()->trace("Joining pipeline thread");
  _pipeline_thread.request_stop();
  _pipeline_thread.join();
  pc::logger()->trace("Pipeline thread complete");
  bool readied_device_memory = _device_memory_ready.load();
  pc::logger()->trace("Need to free gpu device memory: {}",
                      readied_device_memory ? "yes" : "no");
  if (readied_device_memory) {
    free_device_memory();
    pc::logger()->trace("Device memory freed");
  }
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
    pc::logger()->trace("creating new ob::Pipeline");
    ob::Pipeline pipeline{ob_device};

    auto ob_config = std::make_shared<ob::Config>();

    auto &device_config = std::get<OrbbecDeviceConfiguration>(_config);

    const auto [colour_width, colour_height] =
        orbbec::resolution(device_config.color_resolution);
    const auto [depth_width, depth_height] =
        orbbec::resolution(device_config.depth_resolution);

    auto colour_profile_list = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
    // TODO enable without colour too
    if (!colour_profile_list) {
      pc::logger()->error("No Orbbec colour profiles available");
      set_error_state(true);
      return;
    }
    std::shared_ptr<ob::VideoStreamProfile> colour_profile;
    try {
      colour_profile = colour_profile_list->getVideoStreamProfile(
          colour_width, colour_height, OB_FORMAT_RGB, OB_FPS_ANY);
    } catch (const ob::Error &e) {
      pc::logger()->error("Failed to get colour profile: {}", e.what());
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
      pc::logger()->warn("Failed to get HW D2C depth profile list: {}",
                         e.what());
    }

    if (depth_d2c_list && depth_d2c_list->count() > 0) {
      try {
        depth_profile = depth_d2c_list->getVideoStreamProfile(
            depth_width, depth_height, OB_FORMAT_Y16, OB_FPS_ANY);
        ob_config->setAlignMode(ALIGN_D2C_HW_MODE);
      } catch (const ob::Error &e) {
        pc::logger()->error("Failed to select HW D2C depth profile: {}",
                            e.what());
      }
    }

    // fallback to software D2C if HW failed or unsupported for this combo
    if (!depth_profile) {
      pc::logger()->warn("Falling back to ALIGN_D2C_SW_MODE");
      try {
        auto depth_sw_list =
            pipeline.getD2CDepthProfileList(colour_profile, ALIGN_D2C_SW_MODE);
        if (!depth_sw_list || depth_sw_list->count() == 0) {
          pc::logger()->error("No SW D2C depth profiles available");
          set_error_state(true);
          return;
        }
        depth_profile = depth_sw_list->getVideoStreamProfile(
            depth_width, depth_height, OB_FORMAT_Y16, OB_FPS_ANY);
        ob_config->setAlignMode(ALIGN_D2C_SW_MODE);
      } catch (const ob::Error &e) {
        pc::logger()->error("Failed to select SW D2C depth profile: {}",
                            e.what());
        set_error_state(true);
        return;
      }
    }

    const auto fps =
        std::min(depth_profile->getFps(), colour_profile->getFps());
    device_config.fps.set(fps);
    update_config(device_config);

    pc::logger()->debug("stream at: {} fps", fps);

    pc::logger()->trace("enabling stream");

    ob_config->enableStream(depth_profile);
    ob_config->setFrameAggregateOutputMode(
        OB_FRAME_AGGREGATE_OUTPUT_ALL_TYPE_FRAME_REQUIRE);

    // this is frame sync between the devices own depth and colour cameras
    pipeline.enableFrameSync();

    try {
      pc::logger()->trace("attempting to start pipeline");
      pipeline.start(ob_config);
    } catch (const ob::Error &e) {
      pc::logger()->error("Failed to start Orbbec pipeline: {}", e.what());
      set_error_state(true);
      return;
    }

    auto ob_camera_parameters = pipeline.getCameraParam();
    auto ob_calibration_parameters = pipeline.getCalibrationParam(ob_config);

    // // the xy_table converts depths at certain indices to
    // uint32_t xy_table_size = depth_width * depth_height * 2;
    // std::vector<float> xy_table_data(xy_table_size);
    // OBXYTables ob_xy_tables{};
    // if (device_config.conversion_mode ==
    //     OrbbecDeviceConfiguration::PointConversionMode::Custom) {
    //   pc::logger()->trace("Initialising transformation tables");
    //   bool table_init =
    //       ob::CoordinateTransformHelper::transformationInitXYTables(
    //           ob_calibration_parameters, OB_SENSOR_DEPTH,
    //           xy_table_data.data(), &xy_table_size, &ob_xy_tables);
    //   if (!table_init) {
    //     pc::logger()->error("Failed to initialise XY transformation tables. "
    //                         "Switching to OrbbecSDK conversion.");
    //     // TODO is this thread safe? or a race condition
    //     device_config.conversion_mode =
    //         OrbbecDeviceConfiguration::PointConversionMode::OrbbecSDK;
    //   }
    // }

    ob::PointCloudFilter point_cloud_filter;
    point_cloud_filter.setCameraParam(ob_camera_parameters);

    // TODO: query actual depth unit scale from device
    constexpr float depth_position_scale = 1.0f;
    point_cloud_filter.setPositionDataScaled(depth_position_scale);
    point_cloud_filter.setCreatePointFormat(OB_FORMAT_RGB_POINT);

    // create instances to the available backend plugins to transform our
    // point-cloud with

    using Corrade::Containers::Pointer;
    using Corrade::PluginManager::LoadState;

    Pointer<backend::BackendPlugin> cpu_backend;
    Pointer<backend::BackendPlugin> cuda_backend;

    auto &backend_manager = _workspace->backend_plugin_manager;

    for (const auto &plugin : backend_manager->pluginList()) {
      if (backend_manager->loadState(plugin) & LoadState::NotLoaded) continue;
      if (plugin == "CpuBackend") {
        cpu_backend = backend_manager->instantiate(plugin);
        pc::logger()->trace("OrbbecDevice created CPU backend");
        continue;
      }
      if (plugin == "CudaBackend") {
        cuda_backend = backend_manager->instantiate(plugin);
        pc::logger()->trace("OrbbecDevice created CUDA backend");
        continue;
      }
    }

    pc::logger()->trace("Starting process loop for OrbbecDevice {}",
                        device_config.id);

    // processing loop
    while (!stop_token.stop_requested()) {
      std::shared_ptr<ob::FrameSet> frame_set;
      try {
        frame_set = pipeline.waitForFrameset(100);
      } catch (const ob::Error &e) {
        pc::logger()->error("waitForFrameset error: {}", e.what());
        continue;
      }

      if (!frame_set) continue;
      if (_loading_pipeline.load(std::memory_order_relaxed)) set_loading(false);

      auto colour_frame = frame_set->colorFrame();
      auto depth_frame = frame_set->depthFrame();
      if (!colour_frame || !depth_frame) continue;

      {
        ProfilingZone receive_frame_zone("OrbbecDevice::receive_frame");
        receive_frame_zone.text(device_config.id);

        const auto frame_width = colour_frame->width();
        const auto frame_height = colour_frame->height();
        const auto point_count =
            static_cast<size_t>(frame_width * frame_height);

        thread_local static PointCloud point_cloud;
        point_cloud.resize(point_count);

        const auto *ob_depth_frame_ptr =
            reinterpret_cast<const uint16_t *>(depth_frame->getData());
        const auto *ob_color_frame_ptr =
            reinterpret_cast<const color_rgb *>(colour_frame->getData());

        std::span ob_depth_data{ob_depth_frame_ptr, point_count};
        std::span ob_color_data{ob_color_frame_ptr, point_count};

        // we need the colour intrinsic to transform the depth point to colour
        // space
        const auto color_intrinsic =
            ob_calibration_parameters.intrinsics[OB_SENSOR_COLOR];

        // we dont need an extrinsic for the 2d->3d conversion because frame
        // alignment has already been performed within orbbec sdk
        constexpr static OBExtrinsic identity_extrinsic{
            {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
            {0.0f, 0.0f, 0.0f},
        };

        // this is the function that does conversion from 2d orbbec frame data
        // into our pointcaster cloud type in 3d space. it's passed into our
        // transform function to run in parallel using the selected backend
        const auto ob_to_point = [&](int i) -> backend::PointType {
          const auto ob_depth = ob_depth_data[i];
          const auto ob_color = ob_color_data[i];

          const auto x = i % frame_width;
          const auto y = i / frame_height;

          OBPoint3f result;
          ob::CoordinateTransformHelper::transformation2dto3d(
              OBPoint2f(x, y), ob_depth, color_intrinsic, identity_extrinsic,
              &result);

          return {position{.x = static_cast<int16_t>(result.x),
                           .y = -static_cast<int16_t>(result.y),
                           .z = -static_cast<int16_t>(result.z)},
                  color{.r = static_cast<uint8_t>(ob_color.r),
                        .g = static_cast<uint8_t>(ob_color.g),
                        .b = static_cast<uint8_t>(ob_color.b)}};
        };

        bool valid_backend = false;

        switch (device_config.transform.backend.value()) {
        case TransformConfiguration::BackendType::CUDA: {
          if (cuda_backend) {
            cuda_backend->transform_point_cloud(device_config.transform,
                                                point_cloud, ob_to_point);
            valid_backend = true;
          }
          break;
        }
        case TransformConfiguration::BackendType::CPU:
        default: {
          if (cpu_backend) {
            cpu_backend->transform_point_cloud(device_config.transform,
                                               point_cloud, ob_to_point);
            valid_backend = true;
          }
          break;
        }
        }

        if (!valid_backend) {
          pc::logger()->warn(
              "Selected backend for OrbbecDevice could not be loaded");
        }

        bool success = _frame_buffer.try_enqueue(point_cloud);
        if (success) {
          _buffer_updated = true;
          set_updated_time(steady_clock::now());
          notify_point_cloud_updated();
        } else {
          pc::logger()->trace("Dropped frame from "
                              "OrbbecDevice '{}' (ip: {})",
                              device_config.id,
                              device_config.network.ip_address.value());
        }
      }
    }

  } catch (const std::exception &e) {
    pc::logger()->error("Exception in Orbbec processing thread: {}", e.what());
    set_error_state(true);
  } catch (...) {
    pc::logger()->error("Unknown exception in Orbbec processing thread");
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
        auto config = std::get<OrbbecDeviceConfiguration>(this->config());
        pc::logger()->error(
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

const PointCloud &OrbbecDevice::point_cloud() {
  if (_buffer_updated.load(std::memory_order_relaxed)) {
    _frame_buffer.try_dequeue(_current_point_cloud);
    _buffer_updated = false;
  }
  return std::ref(_current_point_cloud);
};

void OrbbecDevice::on_config_field_changed(std::string_view) {
  auto &config = std::get<OrbbecDeviceConfiguration>(_config);

  if (config.network.apply.value()) {
    config.network.apply.set(false);
    set_ip(config.network.ip_address.value(),
           config.network.subnet_mask.value(),
           config.network.gateway_address.value());
  }
}

void OrbbecDevice::set_ip(std::string_view ip_address,
                          std::string_view subnet_mask,
                          std::string_view gateway_address) {
  using pc::util::parse_ip_string;

  auto ip_address_buffer = parse_ip_string(ip_address);
  if (!ip_address_buffer.has_value()) {
    pc::logger()->error("Unable to update IP Address: {}",
                        ip_address_buffer.error());
    return;
  }
  auto subnet_mask_buffer = parse_ip_string(subnet_mask);
  if (!subnet_mask_buffer.has_value()) {
    pc::logger()->error("Unable to update subnet mask: {}",
                        subnet_mask_buffer.error());
    return;
  }
  auto gateway_address_buffer = parse_ip_string(gateway_address);
  if (!subnet_mask_buffer.has_value()) {
    pc::logger()->error("Unable to update gateway address: {}",
                        gateway_address_buffer.error());
    return;
  }

  OBNetIpConfig net_config{};
  net_config.dhcp = 0;
  std::copy(ip_address_buffer->begin(), ip_address_buffer->end(),
            net_config.address);
  std::copy(subnet_mask_buffer->begin(), subnet_mask_buffer->end(),
            net_config.mask);
  std::copy(gateway_address_buffer->begin(), gateway_address_buffer->end(),
            net_config.gateway);

  auto ob_ctx = orbbec_context().wait_til_ready();
  auto &config = std::get<OrbbecDeviceConfiguration>(_config);
  pc::logger()->trace("Forcing IP...");
  auto set_result = ob_ctx->forceIp(config.id.c_str(), net_config);
  if (!set_result) {
    pc::logger()->error("Failed to set network configuration");
  }
  pc::logger()->info(
      "Successfully updated network config for OrbbecDevice '{}'", config.id);
}

// TODO in cuda
bool OrbbecDevice::init_device_memory(std::size_t incoming_point_count) {
  return true;
}

void OrbbecDevice::free_device_memory() {}

} // namespace pc::devices

CORRADE_PLUGIN_REGISTER(OrbbecDevice, pc::devices::OrbbecDevice,
                        "net.pointcaster.DevicePlugin/1.0")
