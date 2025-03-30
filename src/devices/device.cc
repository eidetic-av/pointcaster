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

pc::types::PointCloud synthesized_point_cloud(OperatorList operators) {
  ZoneScopedN("PointCloud::synthesized_point_cloud");

  // TODO

  // can run device->point_cloud(operators) in parallel before using them to
  // construct the result?

  // For the += operator, return PointCloud& instead of PointCloud to avoid
  // copying the result?

  // would it be good to be able to preallocate memory here?

  std::lock_guard lock(devices_access);

  pc::types::PointCloud result{};
  for (const auto& device : attached_devices) {
    result += device->point_cloud();
  }
  
  // TODO the following parallelisation didnt really speed anything up
  // remember to test this with lots of cameras

  // using namespace std::chrono;
  // auto start = high_resolution_clock::now();

  // auto result = tbb::parallel_reduce(
  //     tbb::blocked_range<size_t>(0, attached_devices.size(), 1),
  //     pc::types::PointCloud{},
  //     [&](const tbb::blocked_range<size_t> &r, pc::types::PointCloud pc) {
  //       for (size_t i = r.begin(); i != r.end(); i++) {
  //         pc += attached_devices[i]->point_cloud();
  //       }
  //       return pc;
  //     },
  //     [](pc::types::PointCloud a, pc::types::PointCloud b) { return a += b; });

  // auto end = high_resolution_clock::now();

  // pc::logger->info("parallel took: {}ms",
  //                  duration_cast<milliseconds>(end - start).count());

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
