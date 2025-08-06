#include "device.h"

#include "../gui/widgets.h"
#include "../logger.h"
#include "../operators/session_operator_host.h"
#include "../parameters.h"
#include "../path.h"
#include "../profiling.h"
#include "../string_map.h"
#include "k4a/k4a_config.gen.h"
#include "k4a/k4a_device.h"
#include "k4a/k4a_driver.h"
#include "orbbec/orbbec_device.h"
#include "orbbec/orbbec_device_config.gen.h"
#include "sequence/ply_sequence_player.h"
#include "sequence/ply_sequence_player_config.gen.h"
#include <chrono>
#include <imgui.h>
#include <k4abttypes.h>
#include <memory>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#ifndef __CUDACC__
#include <zpp_bits.h>
#endif

namespace pc::devices {

using pc::gui::slider;
using pc::operators::OperatorList;

namespace {
// this cache makes sure the pointcloud for each session is only computed once,
// synthesized_point_cloud then returns pre-computed frames if the caller
// submits the same session arg on before reset_pointcloud_frames is called
pc::string_map<types::PointCloud> pointcloud_frame_cache;
} // namespace

namespace detail {

void declare_device_parameters(DeviceConfigurationVariant &config_variant) {
  std::visit(
      [](auto &device_config) {
        using T = std::decay_t<decltype(device_config)>;
        session_label_from_id[T::PublishPath] = T::PublishPath;
        pc::parameters::declare_parameters(T::PublishPath, device_config.id,
                                           device_config);
      },
      config_variant);
}

void unbind_device_parameters(
    const DeviceConfigurationVariant &config_variant) {
  std::visit(
      [](const auto &device_config) {
        pc::parameters::unbind_parameters(device_config.id);
      },
      config_variant);
}

} // namespace detail

void reset_pointcloud_frames() { pointcloud_frame_cache.clear(); }

pc::types::PointCloud &compute_or_get_point_cloud(
    std::string_view session_id, std::map<std::string, bool> active_devices_map,
    std::vector<operators::OperatorConfigurationVariant> &session_operators) {
  // if we've already computed a pointcloud for the target session this frame,
  // return that
  if (pointcloud_frame_cache.contains(session_id)) {
    return pointcloud_frame_cache[session_id];
  }

  using namespace pc::profiling;
  ProfilingZone point_cloud_zone("PointCloud::compute_or_get_point_cloud");

  pointcloud_frame_cache[session_id] = {};
  auto& pointcloud = pointcloud_frame_cache[session_id];
  {

    std::lock_guard device_lock(devices_access);
    std::lock_guard config_lock(device_configs_access);

    for (size_t i = 0; i < attached_devices.size(); i++) {
      auto &device = attached_devices[i];
      std::visit(
          [&](auto &&config) {
            using T = std::decay_t<decltype(config)>;
            if (!active_devices_map.contains(config.id) ||
                active_devices_map.at(config.id) == false) {
              return;
            }
            auto device_cloud = device->point_cloud();
            {
              ProfilingZone combine_zone(std::format("{} operator+=", T::Name));
              pointcloud += std::move(device_cloud);
            }
          },
          device_configs[i].get());
    }
  }

  {
    ProfilingZone apply_zone("Apply operators");
    pointcloud =
        pc::operators::apply(pointcloud, session_operators, session_id);
  }
  return pointcloud;
}

pc::types::PointCloud
synthesized_point_cloud(OperatorList &operators,
                        const std::string_view session_id) {
  using namespace pc::profiling;
  ProfilingZone point_cloud_zone("PointCloud::synthesized_point_cloud");

  // TODO

  // can run device->point_cloud(operators) in parallel using a thread pool
  // before construct the full result of combined devices?

  // For the += operator, return PointCloud& instead of PointCloud to avoid
  // copying the result?

  // would it be good to be able to preallocate and/or reuse memory here?
  pc::types::PointCloud result{};
  {

    std::lock_guard device_lock(devices_access);
    std::lock_guard config_lock(device_configs_access);
    point_cloud_zone.text("Acquired device & device_config locks");

    for (size_t i = 0; i < attached_devices.size(); i++) {
      auto &device = attached_devices[i];
      std::visit(
          [&](auto &&config) {
            using T = std::decay_t<decltype(config)>;

            // TODO make a generic way to check if this device is part
            // of the static channel or the live channel
            if constexpr (std::same_as<T, PlySequencePlayerConfiguration>) {
              if (config.streaming.use_static_channel) return;
            }

            types::PointCloud device_cloud;
            {
              ProfilingZone get_zone("Compute pointcloud");
              device_cloud = device->point_cloud();
            }
            {
              ProfilingZone combine_zone(std::format("{} operator+=", T::Name));
              result += std::move(device_cloud);
            }
          },
          device_configs[i].get());
    }
  }

  {
    ProfilingZone apply_zone("Apply operators");
    result = pc::operators::apply(result, operators, session_id);
  }

  return result;
}

std::vector<StaticPointcloudRef>
static_point_clouds(OperatorList &operators,
                        const std::string_view session_id) {
  // TODO limit to session_id

  using namespace pc::profiling;
  ProfilingZone point_cloud_zone("PointCloud::static_point_clouds");
  std::vector<StaticPointcloudRef> results{};
  {
    std::lock_guard device_lock(devices_access);
    std::lock_guard config_lock(device_configs_access);

    for (size_t i = 0; i < attached_devices.size(); i++) {
      auto &device = attached_devices[i];
      std::visit(
          [&](auto &&config) {
            using T = std::decay_t<decltype(config)>;
            // TODO make a generic way to check if this device is part
            // of the static channel or the live channel
            if constexpr (std::same_as<T, PlySequencePlayerConfiguration>) {

              bool send_empty_frame = false;

              static std::unordered_map<std::string, bool> was_static_last_frame;
              bool was_static = was_static_last_frame[config.id];
              bool now_static = config.streaming.use_static_channel;
              was_static_last_frame[config.id] = now_static;
              if (was_static && !now_static) {
                send_empty_frame = true;
              }

              if (now_static) {
                static std::unordered_map<std::string, bool>
                    was_active_last_frame;
                bool was_active = was_active_last_frame[config.id];
                bool now_active = config.active;
                was_active_last_frame[config.id] = now_active;
                if (was_active && !now_active) {
                  send_empty_frame = true;
                } else if (!was_active && now_active) {
                  config.streaming.requires_update = true;
                }
              }

              if (send_empty_frame) {
                constexpr auto max_s = std::numeric_limits<short>::max();
                // serialization fails if there isn't at least one point in the point cloud
                static PointCloud empty_point_cloud{
                    .positions = {{max_s, max_s, max_s}},
                    .colors = {{0, 0, 0}}};
                config.streaming.requires_update = true;
                results.push_back({.source_id = config.id,
                                   .requires_update = std::ref(
                                       config.streaming.requires_update),
                                   .point_cloud = std::ref(empty_point_cloud),
                                   .operators = operators,
                                   .disable_compression =
                                       config.streaming.disable_compression});
                pc::logger->info("pushed empty");
              } else if (now_static) {
                auto player =
                    std::static_pointer_cast<PlySequencePlayer>(device);
                if (!player->last_point_cloud().has_value()) return;
                results.push_back(
                    {.source_id = config.id,
                     .requires_update =
                         std::ref(config.streaming.requires_update),
                     .point_cloud = player->last_point_cloud().value(),
                     .operators = operators,
                     .disable_compression =
                         config.streaming.disable_compression});
                if (config.streaming.requires_update) {
                  pc::logger->info("pushed real");
                }
              }
            }
          },
          device_configs[i].get());
    }
  }

  return results;
}

void send_all_static_point_clouds() {
  std::lock_guard config_lock(device_configs_access);
  for (auto &device : PlySequencePlayer::attached_devices) {
    auto &config = device.get().config();
    if (config.active && config.streaming.use_static_channel) {
      config.streaming.requires_update = true;
    }
  };
}

// TODO all of this skeleton stuff needs to be made generic accross multiple
// camera types
std::vector<K4ASkeleton> scene_skeletons() {
  std::vector<K4ASkeleton> result;
  // for (auto& device : pc::devices::attached_devices) {
  // 	auto driver = dynamic_cast<K4ADriver*>(device->_driver.get());
  // 	if (!driver->tracking_bodies()) continue;
  // 	for (auto& skeleton : driver->skeletons()) {
  // 		result.push_back(skeleton);
  // 	}
  // }
  return result;
}

} // namespace pc::devices
