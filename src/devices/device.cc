#include "device.h"

#include "../gui/widgets.h"
#include "../logger.h"
#include "../operators/session_operator_host.h"
#include "../parameters.h"
#include "../path.h"
#include "../profiling.h"
#include "../string_map.h"
#include "k4a/k4a_device.h"
#include "k4a/k4a_driver.h"
#include "orbbec/orbbec_device.h"
#include "sequence/ply_sequence_player.h"
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

namespace detail {

void declare_device_parameters(DeviceConfigurationVariant &config_variant) {
  std::visit(
      [](auto &device_config) {
        pc::parameters::declare_parameters(device_config.id, device_config);
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

// Device::Device(DeviceConfiguration config) : _config(config) {};

using pc::gui::slider;
using pc::operators::OperatorList;

namespace {
// this cache makes sure the pointcloud for each session is only computed once,
// synthesized_point_cloud then returns pre-computed frames if the caller
// submits the same session arg on before reset_pointcloud_frames is called
pc::string_map<types::PointCloud> pointcloud_frame_cache;
} // namespace

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

    // TODO apply operators using per-device lists
    // result += device->point_cloud(config.operators);
    // instead of the following empty_list
    static std::vector<operators::OperatorHostConfiguration> empty_list{};

    for (size_t i = 0; i < attached_devices.size(); i++) {
      auto &device = attached_devices[i];
      std::visit(
          [&](auto &&config) {
            using T = std::decay_t<decltype(config)>;
            if (!active_devices_map.contains(config.id) ||
                active_devices_map.at(config.id) == false) {
              return;
            }
            auto device_cloud = device->point_cloud(empty_list);
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
    pointcloud = pc::operators::apply(pointcloud, session_operators);
  }
  return pointcloud;
}

pc::types::PointCloud synthesized_point_cloud(OperatorList &operators) {
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

    // TODO apply operators using per-device lists
    // result += device->point_cloud(config.operators);
    // instead of the following empty_list
    static std::vector<operators::OperatorHostConfiguration> empty_list{};

    for (size_t i = 0; i < attached_devices.size(); i++) {
      auto &device = attached_devices[i];
      std::visit(
          [&](auto &&config) {
            using T = std::decay_t<decltype(config)>;
            auto device_cloud = device->point_cloud(empty_list);
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
    result = pc::operators::apply(result, operators);
  }

  return result;
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

// void Device::draw_imgui_controls() {
// 	ImGui::PushID(_driver->id().c_str());

// 	auto open = _driver->is_open();
// 	if (!open) ImGui::BeginDisabled();

// 	draw_device_controls();
// 	if (!open) ImGui::EndDisabled();

// 	ImGui::PopID();
// };

pc::types::position global_translate;
void draw_global_controls() {
  ImGui::SetNextWindowPos({150.0f, 50.0f}, ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.8f);
  ImGui::PushID("GlobalTransform");
  ImGui::Begin("Global Transform", nullptr);
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.8f);
  ImGui::PopItemWidth();
  ImGui::End();
  ImGui::PopID();
}

} // namespace pc::devices
