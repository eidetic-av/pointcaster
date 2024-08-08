#include "session_operator_host.h"
#include "../gui/catpuccin.h"
#include "../gui/widgets.h"
#include "../logger.h"
#include "../parameters.h"
#include "../string_utils.h"
#include "../uuid.h"
#include "../aabb.h"
#include "noise_operator.gen.h"
#include "rake_operator.gen.h"
#include "range_filter_operator.gen.h"
#include "rotate_operator.gen.h"
#include "sample_filter_operator.gen.h"
#include "session_bounding_boxes.h"
#include <algorithm>
#include <execution>
#include <functional>
#include <optional>

#include <tbb/tbb.h>

namespace pc::operators {

	using namespace pc::types;
	using namespace pc::parameters;
	using namespace oneapi;

	SessionOperatorHost::SessionOperatorHost(
		OperatorHostConfiguration& config, Scene3D& scene,
		Magnum::SceneGraph::DrawableGroup3D& parent_group)
		: _config(config), _scene(scene), _parent_group(parent_group) {

		// for loading an existing list of operators
		for (auto& operator_instance : _config.operators) {
			// we need to visit each possible operator variant
			std::visit(
				[&](auto&& operator_config) {
					// and declare it's saved ID with its parameters
					declare_parameters(std::to_string(operator_config.id),
					operator_config);

			using T = std::decay_t<decltype(operator_config)>;

			if constexpr (std::is_same_v<T, RangeFilterOperatorConfiguration>) {
				RangeFilterOperator::init(operator_config, _scene, _parent_group,
					next_bounding_box_color());
			}

				},
				operator_instance);
		}

		using namespace std::chrono_literals;


		instance = this;
	};

	bool operator_collapsing_header(
		const char* label,
		std::optional<std::function<void()>> delete_callback = {}) {
		using namespace ImGui;
		PushID(gui::_parameter_index++);
		PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
		bool* p_open = GetStateStorage()->GetBoolRef(GetID(label), false);
		ImGuiStyle& style = ImGui::GetStyle();
		if (ArrowButton(fmt::format("{}##arrow", label).c_str(),
			*p_open ? ImGuiDir_Down : ImGuiDir_Right)) {
			*p_open ^= 1;
		}
		ImGui::SameLine();
		if (delete_callback.has_value()) {
			const auto close_button_width = ImGui::CalcTextSize(" X ").x * 1.5;
			if (Button(label, ImVec2(-FLT_MIN - close_button_width, 0.0f))) {
				*p_open ^= 1;
			}
			ImGui::SameLine();
			if (ImGui::Button(" X ")) {
				delete_callback.value()();
			};
		}
		else {
			if (Button(label, ImVec2(-FLT_MIN, 0.0f))) {
				*p_open ^= 1;
			}
		}
		PopStyleVar();
		ImGui::PopID();
		return *p_open;
	}

	void SessionOperatorHost::draw_gizmos() {

		operator_bounding_boxes.clear();

		for (const auto& operator_config : _config.operators) {
			std::visit([this](auto&& config) {
				using T = std::decay_t<decltype(config)>;

				if constexpr (std::is_same_v <T, pcl_cpu::ClusterExtractionConfiguration>) {
					auto& pipeline = pcl_cpu::ClusterExtractionPipeline::instance();

					if (config.draw_voxels) {
						auto current_voxels = pipeline.current_voxels.load();
						if (current_voxels.get() != nullptr) {

							// TODO this block is too slow

							//using namespace std::chrono;
							//using namespace std::chrono_literals;
							//auto start = high_resolution_clock::now();
							auto voxel_count = current_voxels->size();								// pc::logger->debug("voxel count: {}", voxel_count);
							for (int i = 0; i < voxel_count; i++) {
								pcl::PointXYZ p = current_voxels->points[i];
								// mm to metres
								set_voxel(i, { p.x / 1000.0f, p.y / 1000.0f, p.z / 1000.0f });
							}
							//auto end = high_resolution_clock::now();
							//int duration_us = duration_cast<microseconds>(end - start).count();
							//float duration_ms = duration_us / 1'000.0f;
							//pc::logger->info("drawing voxels took: {}ms", fmt::format("{:.2f}", duration_ms));
						}
					}

					if (config.draw_clusters) {
						auto latest_clusters_ptr = pipeline.current_clusters.load();
						if (latest_clusters_ptr.get() != nullptr) {
							//using namespace std::chrono;
							//using namespace std::chrono_literals;
							//auto start = high_resolution_clock::now();
							auto& latest_clusters = *latest_clusters_ptr;
							auto cluster_count = latest_clusters.size();
							// pc::logger->debug("cluster count: {}", cluster_count);
							for (int i = 0; i < cluster_count; i++) {
								pc::AABB aabb = latest_clusters[i];
								auto position = aabb.center() / 1000.0f; // mm to metres
								auto size = aabb.extents() / 1000.0f;
								set_cluster(i, position, size);
							}
							//auto end = high_resolution_clock::now();
							//int duration_us = duration_cast<microseconds>(end - start).count();
							//float duration_ms = duration_us / 1'000.0f;
							//pc::logger->info("drawing clusters took: {}ms", fmt::format("{:.2f}", duration_ms));
						}
					}

				}

			}, operator_config);

		}


	}

	void SessionOperatorHost::draw_imgui_window() {
		ImGui::SetNextWindowSize({ 600, 400 }, ImGuiCond_FirstUseEver);
		ImGui::Begin("Session Operators", nullptr);

		static bool select_node = false;

		if (ImGui::Button("Add session operator")) {
			ImGui::OpenPopup("Add session operator");
		}
		ImGui::Spacing();

		if (ImGui::BeginPopup("Add session operator")) {
			// populate menu with all Operator types

			apply_to_all_operators([this](auto&& operator_type) {
				using T = std::remove_reference_t<decltype(operator_type)>;
				if (ImGui::Selectable(T::Name)) {

					auto operator_config = T();
					operator_config.id = pc::uuid::digit();
					// create a new instance of our operator configuration and add it to our
					// session operator list
					auto& variant_ref = _config.operators.emplace_back(operator_config);

					// declare the instance as parameters to bind to this new operator's id
					auto& config_instance = std::get<T>(variant_ref);
					declare_parameters(std::to_string(operator_config.id), config_instance);

					if constexpr (std::is_same_v<T, RangeFilterOperatorConfiguration>) {
						RangeFilterOperator::init(operator_config, _scene, _parent_group,
							next_bounding_box_color());
					}

					ImGui::CloseCurrentPopup();
				}
				});

			ImGui::EndPopup();
		}

		std::optional<uid> marked_for_delete;

		for (auto& operator_config : _config.operators) {
			ImGui::PushID(gui::_parameter_index++);

			std::visit(
				[&](auto&& config) {
					using T = std::decay_t<decltype(config)>;

					auto id = std::to_string(gui::_parameter_index++);
					ImGui::PushID(gui::_parameter_index++);
					ImGui::BeginGroup();
					if (operator_collapsing_header(
						pc::strings::concat(T::Name, "##" + id).c_str(),
						[&] { marked_for_delete = config.id; })) {
						config.unfolded = true;

						if (pc::gui::draw_parameters(config.id)) {
							if constexpr (std::is_same_v<T,
								RangeFilterOperatorConfiguration>) {
								RangeFilterOperator::init(config, _scene, _parent_group,
									next_bounding_box_color());
							}
						}
					}
					else {
						config.unfolded = false;
					}
					ImGui::EndGroup();
					ImGui::PopID();
				},
				operator_config);
			ImGui::PopID();
		}

		if (marked_for_delete.has_value()) {
			const auto check_marked_id = [&](const auto& config) {
				return config.id == marked_for_delete.value();
			};
			auto new_end = std::remove_if(
				_config.operators.begin(), _config.operators.end(),
				[&](const auto& op) { return std::visit(check_marked_id, op); });
			_config.operators.erase(new_end, _config.operators.end());
		}

		ImGui::End();
	}

	void SessionOperatorHost::set_voxel(pc::types::Float3 position, pc::types::Float3 size) {
		set_or_create_bounding_box(pc::uuid::digit(), size, position, _scene, _parent_group, catpuccin::magnum::mocha_blue);
	}

	void SessionOperatorHost::set_voxel(uid id, pc::types::Float3 position, pc::types::Float3 size) {
		set_or_create_bounding_box(id, size, position, _scene, _parent_group, catpuccin::magnum::mocha_blue);
	}

	void SessionOperatorHost::set_cluster(uid id, pc::types::Float3 position, pc::types::Float3 size) {
		//auto color_idx = id % bounding_box_colors.size();
		//auto& color = bounding_box_colors[color_idx];
		//set_or_create_bounding_box(10000 + id, size, position, _scene, _parent_group, color);
		set_or_create_bounding_box(10000 + id, size, position, _scene, _parent_group, catpuccin::magnum::mocha_red);
	}

} // namespace pc::operators
