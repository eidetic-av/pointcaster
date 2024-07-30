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
#include <functional>
#include <optional>

#include <tbb/tbb.h>

#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>

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

		pc::logger->info("Initialising TBB PCL Pipeline (default_concurrency = {})",
			tbb::info::default_concurrency());

		using namespace std::chrono_literals;

		_pcl_thread = std::jthread([this]() {
			tbb::parallel_pipeline(
				tbb::info::default_concurrency(),
				tbb::make_filter<void, pcl_input_frame*>(
					tbb::filter_mode::serial_in_order, pcl_ingest{pcl_queue, _shutdown_pipeline}) &
				tbb::make_filter<pcl_input_frame*, void>(
					tbb::filter_mode::parallel, pcl_process{latest_voxels, latest_clusters})
			);
		});

		instance = this;
	};


	SessionOperatorHost::pcl_input_frame* SessionOperatorHost::pcl_ingest::operator()(tbb::flow_control& fc) const {
		using namespace std::chrono_literals;
		auto* positions = new pcl_input_frame;
		while (!queue.try_pop(*positions)) {
			if (shutdown_pipeline) {
				delete positions;
				fc.stop();
				return nullptr;
			}
			std::this_thread::sleep_for(2ms);
		}
		return positions;
	}

	void SessionOperatorHost::pcl_process::operator()(SessionOperatorHost::pcl_input_frame* frame) const {
		if (frame) {

			// TODO check timestamp against current time

			// TODO configuration options for voxel sampling and clustering

			auto& positions = frame->positions;
			auto count = positions.size();

			using namespace std::chrono;
			using namespace std::chrono_literals;

			auto start_time = system_clock::now();

			//  convert position type into PCL structures
			auto pcl_positions = thrust::host_vector<pcl::PointXYZ>(count);
			thrust::transform(positions.begin(), positions.end(), pcl_positions.begin(), PositionToPointXYZ());

			// and copy the transformed result into a PCL PointCloud object
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
			cloud->points.resize(count);
			std::copy(pcl_positions.begin(), pcl_positions.end(), cloud->points.begin());

			// filter the cloud into a voxel grid
			pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
			voxel_grid.setInputCloud(cloud);
			voxel_grid.setLeafSize(200, 200, 200); // mm
			voxel_grid.filter(*filtered_cloud);

			latest_voxels.store(filtered_cloud);

			auto voxelisation_time = system_clock::now();
			auto voxelisation_us = duration_cast<microseconds>(voxelisation_time - start_time);
			auto voxelisation_ms = voxelisation_us.count() / 1000.0f;
			pc::logger->debug("voxelisation time: {:.2f}ms", voxelisation_ms);

			// create kdtree
			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
			tree->setInputCloud(filtered_cloud);

			auto kdgen_time = system_clock::now();
			auto kdgen_us = duration_cast<microseconds>(kdgen_time - voxelisation_time);
			auto kdgen_ms = kdgen_us.count() / 1000.0f;
			pc::logger->debug("kdgen time: {:.2f}ms", kdgen_ms);

			// perform clustering
			std::vector<pcl::PointIndices> cluster_indices;
			pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
			ec.setClusterTolerance(270); // mm
			ec.setMinClusterSize(10); // voxel counts per cluster generated
			ec.setMaxClusterSize(100);
			ec.setSearchMethod(tree);
			ec.setInputCloud(filtered_cloud);
			ec.extract(cluster_indices);

			auto cluster_time = system_clock::now();
			auto cluster_us = duration_cast<microseconds>(cluster_time - kdgen_time);
			auto cluster_ms = cluster_us.count() / 1000.0f;
			pc::logger->debug("cluster time: {:.2f}ms", cluster_ms);

			// calculate bounding boxes
			tbb::concurrent_vector<pc::AABB> cluster_bounds;
			tbb::parallel_for(tbb::blocked_range<size_t>(0, cluster_indices.size()),
				[&cluster_indices, &filtered_cloud, &cluster_bounds](const tbb::blocked_range<size_t>& r) {
					for (size_t i = r.begin(); i != r.end(); ++i) {
						const auto& indices = cluster_indices[i];

						pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
						for (const auto& index : indices.indices) {
							cloud_cluster->points.push_back(filtered_cloud->points[index]);
						}
						cloud_cluster->width = cloud_cluster->points.size();
						cloud_cluster->height = 1;
						cloud_cluster->is_dense = true;

						// Compute the bounding box
						pcl::PointXYZ min_pt, max_pt;
						pcl::getMinMax3D(*cloud_cluster, min_pt, max_pt);

						pc::types::Float3 min_pt_f(min_pt.x, min_pt.y, min_pt.z);
						pc::types::Float3 max_pt_f(max_pt.x, max_pt.y, max_pt.z);

						cluster_bounds.push_back(pc::AABB(min_pt_f, max_pt_f));
					}
				});

			auto aabb_time = system_clock::now();
			auto aabb_us = duration_cast<microseconds>(aabb_time - cluster_time);
			auto aabb_ms = aabb_us.count() / 1000.0f;
			pc::logger->debug("aabb time: {:.2f}ms", aabb_ms);

			// copy data to host instance
			auto output_bounds = std::make_shared<std::vector<pc::AABB>>(cluster_bounds.size());
			std::copy(cluster_bounds.begin(), cluster_bounds.end(), output_bounds->begin());

			latest_clusters.store(output_bounds);

			delete frame;
		}
	}

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

	void SessionOperatorHost::set_voxel(uid id, pc::types::Float3 position, pc::types::Float3 size) {
		//auto color_idx = id % bounding_box_colors.size();
		//auto& color = bounding_box_colors[color_idx];
		set_or_create_bounding_box(id, size, position, _scene, _parent_group, catpuccin::magnum::mocha_blue);
	}

	void SessionOperatorHost::set_cluster(uid id, pc::types::Float3 position, pc::types::Float3 size) {
		set_or_create_bounding_box(10000 + id, size, position, _scene, _parent_group, catpuccin::magnum::mocha_red);
	}

} // namespace pc::operators
