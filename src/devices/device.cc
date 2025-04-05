#include "device.h"

#include "../gui/widgets.h"
#include "../logger.h"
#include "../operators/session_operator_host.h"
#include "../parameters.h"
#include "../path.h"
#include "k4a/k4a_device.h"
#include "k4a/k4a_driver.h"
#include "orbbec/orbbec_device.h"
#include "sequence/ply_sequence_player.h"

#include <imgui.h>
#include <k4abttypes.h>
#include <memory>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tracy/Tracy.hpp>

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

using pc::gui::slider;
using pc::operators::OperatorList;

// Device::Device(DeviceConfiguration config) : _config(config) {};

pc::types::PointCloud synthesized_point_cloud(OperatorList& operators) {
  ZoneScopedN("PointCloud::synthesized_point_cloud");

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

    static std::vector<operators::OperatorHostConfiguration> empty_list{};

    for (size_t i = 0; i < attached_devices.size(); i++) {
      auto& device = attached_devices[i];
      std::visit(
          [&](auto &&config) {
            // result += device->point_cloud(config.operators);
            result += device->point_cloud(empty_list);
          },
          device_configs[i].get());
    }
  }

  return pc::operators::apply(result, operators);
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
		ImGui::SetNextWindowPos({ 150.0f, 50.0f }, ImGuiCond_FirstUseEver);
		ImGui::SetNextWindowBgAlpha(0.8f);
		ImGui::PushID("GlobalTransform");
		ImGui::Begin("Global Transform", nullptr);
		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.8f);
		ImGui::PopItemWidth();
		ImGui::End();
		ImGui::PopID();
	}

} // namespace pc::devices
