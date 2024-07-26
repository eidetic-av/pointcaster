#include <cmath>
#include <imgui.h>
#include <imnodes.h>
#include <fmt/format.h>
#include "operator_graph.h"
#include "../gui/widgets.h"
#include "../parameters.h"
#include "../string_utils.h"
#include "../gui/catpuccin.h"
#include "../devices/device_config.gen.h"
#include "../camera/camera_config.gen.h"

namespace pc::graph {

	using namespace operators;
	using namespace devices;
	using namespace camera;

	OperatorGraph::OperatorGraph(std::string_view id) : _id(id) {
		ImNodes::CreateContext();
	}

	OperatorGraph::~OperatorGraph() {
		ImNodes::DestroyContext();
	}

	void OperatorGraph::add_input_node(NodeDataVariant node_data) {
		_nodes.emplace_back(pc::uuid::digit(), LinkDirection::Out, node_data);
	}

	void OperatorGraph::add_output_node(NodeDataVariant node_data) {
		_nodes.emplace_back(pc::uuid::digit(), LinkDirection::In, node_data);
	}

	void OperatorGraph::remove_node(const std::string& id) {
		// _nodes.erase(id);
	}

	void OperatorGraph::draw() {
		ImGui::SetNextWindowSize({ 600, 400 }, ImGuiCond_FirstUseEver);

		if (ImGui::Begin(_id.c_str(), nullptr)) {

			ImNodes::BeginNodeEditor();

			int attribute_id = 1;
			const bool check_for_key_events = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && ImNodes::IsEditorHovered();
			bool delete_selection = false;
			static bool nodes_selected = false;

			// space bar or right click opens the create new node menu
			const auto new_node_position = ImGui::GetMousePos();
			const auto new_node_popup = fmt::format("Add Node##{}", _id);
			if (check_for_key_events &&
				(ImGui::IsKeyReleased(ImGuiKey_Space) || ImGui::IsMouseClicked(ImGuiMouseButton_Right))) {
				ImGui::OpenPopup(new_node_popup.c_str());
			}
			if (ImGui::BeginPopup(new_node_popup.c_str())) {
				ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 2, 2 });
				ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, { 2, 2 });
				using namespace operators;
				// populate menu with all Operator types
				apply_to_all_operators([&](auto&& operator_type) {
					using T = std::remove_reference_t<decltype(operator_type)>;

					if (ImGui::Selectable(T::Name)) {
						// instantiate a new operator configuration in the new node
						const unsigned long node_id = pc::uuid::digit();
						auto operator_config = T{ .id = node_id };
						ImNodes::SetNodeScreenSpacePos(node_id, new_node_position);
						_nodes.emplace_back(node_id, LinkDirection::InOut, operator_config);

						// declare the new operator's parameters to draw them inside the node
						auto& new_config = std::get<OperatorConfigurationVariant>(_nodes.back().node_data);
						std::visit([node_id](auto& operator_config) {
							pc::parameters::declare_parameters(std::to_string(operator_config.id), operator_config);
						}, new_config);

						ImGui::CloseCurrentPopup();
					}
					});
				// add delete option if nodes are selected
				if (nodes_selected) {
					ImGui::Dummy({ 0, 3 });
					if (ImGui::Selectable("Delete##selected")) {
						delete_selection = true;
					};
				}
				ImGui::PopStyleVar(2);
				ImGui::EndPopup();
			}

			// node drawing widgets and utils

			constexpr float node_width = 200.0f;

			constexpr auto draw_node_title = [](std::string_view title) {
				ImNodes::BeginNodeTitleBar();
				ImGui::PushItemWidth(node_width);
				ImGui::Text("%s", title.data());
				ImGui::PopItemWidth();
				ImNodes::EndNodeTitleBar();
			};

			int port_id = -1;
			const auto draw_point_cloud_port = [&](LinkDirection link_direction, auto node_id) {
				if (link_direction == LinkDirection::In) {
					ImNodes::BeginInputAttribute(port_id--, ImNodesPinShape_CircleFilled);
					ImGui::PushItemWidth(node_width);
					ImGui::TextUnformatted("xyzrgb");
					ImGui::PopItemWidth();
					ImNodes::EndInputAttribute();
				}
				else if (link_direction == LinkDirection::Out) {
					ImNodes::BeginOutputAttribute(port_id--, ImNodesPinShape_CircleFilled);
					static const float text_width = ImGui::CalcTextSize("xyzrgb").x;
					ImGui::PushItemWidth(node_width);
					ImGui::Indent(node_width - text_width);
					ImGui::TextUnformatted("xyzrgb");
					ImGui::PopItemWidth();
					ImNodes::EndOutputAttribute();
				}
			};

			for (const auto& node : _nodes)
			{
				ImNodes::BeginNode(node.id);

				auto draw_node = [&](auto&& node_data) {

					using T = std::remove_reference_t<decltype(node_data)>;
					constexpr static auto type_info = serde::type_info<T>;
					const auto member_names = type_info.member_names().members();
					const auto member_defaults = T::Defaults;
					const auto member_min_max_values = T::MinMaxValues;

					draw_node_title(fmt::format("{} ({})", T::Name, node_data.id));

					if (node.link_direction == LinkDirection::In || node.link_direction == LinkDirection::InOut) {
						draw_point_cloud_port(LinkDirection::In, node.id);
					}
					ImGui::Dummy({ 0, 2 });

					// Function to apply on each type member to render it as input or output
					auto handle_member = [&](auto& member_name, auto index) {
						using namespace pc::parameters;
						using MemberType = pc::reflect::type_at_t<typename T::MemberTypes, decltype(index)::value>;

						if constexpr (std::is_arithmetic_v<MemberName>) {
							auto parameter_id = fmt::format("{}.{}", node_data.id, member_name);
							if (!parameter_bindings.contains(parameter_id)) return;

							auto& parameter = parameter_bindings.at(parameter_id);
							auto value = static_cast<MemberType>(get_parameter_value(parameter_id));

							ImNodes::BeginInputAttribute(attribute_id++, ImNodesPinShape_Circle);
							ImGui::Text("%s", member_name.data());
							ImNodes::EndInputAttribute();
							ImGui::SameLine();
							ImGui::Dummy({ 4, 0 });
							ImGui::SameLine();
							ImGui::SetNextItemWidth(100);

							auto& member_min_max = member_min_max_values.at(index);
							auto& member_default = std::get<index>(member_defaults);

							if constexpr (std::is_same_v<int, MemberType>) {
								int& value = std::get<IntReference>(parameter.value).get();
								if (member_min_max.has_value()) {
									const auto& [min_f, max_f] = member_min_max.value();
									const int min = std::rint(min_f);
									const int max = std::rint(max_f);
									const int range = max - min;
									const int speed = std::rint(range / 1000.0f);
									ImGui::DragInt(fmt::format("##{}", parameter_id).c_str(), &value, speed, min, max);
								}
								else {
									ImGui::DragInt(fmt::format("##{}", parameter_id).c_str(), &value);
								}
							}
							else if constexpr (std::is_same_v<float, MemberType>) {
								float& value = std::get<FloatReference>(parameter.value).get();
								if (member_min_max.has_value()) {
									auto& [min, max] = member_min_max.value();
									const float speed = (max - min) / 1000.0f;
									ImGui::DragFloat(fmt::format("##{}", parameter_id).c_str(), &value, speed, min, max);
								}
								else {
									ImGui::DragFloat(fmt::format("##{}", parameter_id).c_str(), &value);
								}
							}
						}

						ImGui::Dummy({ 0, 2 });
					};

					// using an immediately invoked lambda to provide member names and
					// the associated index as a compile-time constant to the handler 
					// function that draws each member on the node
					[&handle_member, &member_names]<std::size_t... Is>(std::index_sequence<Is...>) {
						((handle_member(member_names[Is], std::integral_constant<std::size_t, Is>{})), ...);
					}(std::make_index_sequence<T::MemberCount>{});

					if (node.link_direction == LinkDirection::Out || node.link_direction == LinkDirection::InOut) {
						draw_point_cloud_port(LinkDirection::Out, node.id);
					}
				};

				if (std::holds_alternative<OperatorConfigurationVariant>(node.node_data)) {
					auto& operator_config = std::get<OperatorConfigurationVariant>(node.node_data);
					std::visit(draw_node, operator_config);
				}
				else if (std::holds_alternative<std::reference_wrapper<DeviceConfiguration>>(node.node_data)) {
					auto& device_config = std::get<std::reference_wrapper<DeviceConfiguration>>(node.node_data);
					draw_node(device_config.get());
				}
				else if (std::holds_alternative<std::reference_wrapper<CameraConfiguration>>(node.node_data)) {
					auto& camera_config = std::get<std::reference_wrapper<CameraConfiguration>>(node.node_data);
					draw_node(camera_config.get());
				}

				ImNodes::EndNode();
			}

			// draw connecting links
			for (const auto& link : _links)
			{
				ImNodes::Link(link.id, link.output_id, link.input_id);
			}

			// ImNodes::MiniMap(0.2f, ImNodesMiniMapLocation_BottomLeft);

			// finish rendering the graph
			ImNodes::EndNodeEditor();

			// check for newly created links
			Link new_link;
			if (ImNodes::IsLinkCreated(&new_link.output_id, &new_link.input_id))
			{
				// if both ids multiplied is positive theyre compatible
				// because point cloud ids are negative and attributes are positive
				if (new_link.output_id * new_link.input_id > 0) {
					new_link.id = pc::uuid::digit();
					_links.push_back(new_link);
				};
			}

			// check for links that have been detached
			int detached_link_id;
			if (ImNodes::IsLinkDestroyed(&detached_link_id))
			{
				auto iter = std::find_if(_links.begin(), _links.end(), [&](const auto& link) { return link.id == detached_link_id; });
				_links.erase(iter);
			}

			size_t selected_node_count = ImNodes::NumSelectedNodes();
			nodes_selected = selected_node_count > 0;
			// delete key removes nodes and links
			if (delete_selection || (check_for_key_events && ImGui::IsKeyPressed(ImGuiKey_Delete))) {

				if (selected_node_count > 0) {
					std::vector<int> selected_node_ids(selected_node_count);
					ImNodes::GetSelectedNodes(selected_node_ids.data());
					auto new_end = std::remove_if(_nodes.begin(), _nodes.end(),
						[&selected_node_ids](const Node& node) {
							return std::find(selected_node_ids.begin(), selected_node_ids.end(), node.id) != selected_node_ids.end();
						}
					);
					_nodes.erase(new_end, _nodes.end());
					nodes_selected = 0;
				}

				size_t selected_link_count = ImNodes::NumSelectedLinks();
				if (selected_link_count > 0) {
					std::vector<int> selected_link_ids(selected_link_count);
					ImNodes::GetSelectedLinks(selected_link_ids.data());
					auto end_iter = std::remove_if(_links.begin(), _links.end(),
						[&selected_link_ids](const Link& link) {
							return std::find(selected_link_ids.begin(), selected_link_ids.end(), link.id) != selected_link_ids.end();
						}
					);
					_links.erase(end_iter, _links.end());
				}
			}
		}
		ImGui::End();

		ImGui::ShowIDStackToolWindow();
	}

}