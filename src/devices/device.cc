#include "device.h"
#include "../logger.h"
#include "../operators/session_operator_host.h"
#include "../path.h"
#include "k4a/k4a_driver.h"
#include "orbbec/orbbec_device.h"

#include <imgui.h>
#include <k4abttypes.h>
#include <tracy/Tracy.hpp>

#ifndef __CUDACC__
#include <zpp_bits.h>
#endif

namespace pc::devices {

	using pc::gui::slider;
	using pc::operators::OperatorList;

	Device::Device(DeviceConfiguration config) : _config(config) {};

	pc::types::PointCloud synthesized_point_cloud(OperatorList operators) {
		ZoneScopedN("PointCloud::synthesized_point_cloud");

		// TODO

		// can run device->point_cloud(operators) in parallel before using them to
		// construct the result

		// For the += operator, return PointCloud& instead of PointCloud to avoid
		// copying the result.

		// it would be good to be able to preallocate memory here?

		// TODO
		// theres probs a way to run the filtering operators while doing
		// device->point_cloud() so the copies of the resulting data are less
		// expensive, then only run the updates on "apply"

		pc::types::PointCloud result{};

		{
			std::lock_guard<std::mutex> lock(Device::devices_access);
			for (auto& device : Device::attached_devices) {
				{
					ZoneScopedN("k4a operators & add result");
					result += device->point_cloud();
				}
			}
		}

		{
			std::lock_guard lock(OrbbecDevice::devices_access);
			int i = 1;
			for (auto& device : OrbbecDevice::attached_devices) {
				ZoneScopedN("orbbec operators & add result");
				result += device.get().point_cloud();
			}
		}

		result = pc::operators::apply(result, operators);

		return result;
	}

	// TODO all of this skeleton stuff needs to be made generic accross multiple
	// camera types
	std::vector<K4ASkeleton> scene_skeletons() {
		std::vector<K4ASkeleton> result;
		for (auto& device : Device::attached_devices) {
			auto driver = dynamic_cast<K4ADriver*>(device->_driver.get());
			if (!driver->tracking_bodies()) continue;
			for (auto& skeleton : driver->skeletons()) {
				result.push_back(skeleton);
			}
		}
		return result;
	}

	void Device::draw_imgui_controls() {
		ImGui::PushID(_driver->id().c_str());

		auto open = _driver->is_open();
		if (!open) ImGui::BeginDisabled();

		draw_device_controls();
		if (!open) ImGui::EndDisabled();

		ImGui::PopID();
	};

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
