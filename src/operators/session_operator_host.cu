#include <future>
#include <thread>
#include <variant>
#include <deque>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <tracy/Tracy.hpp>
#include "../logger.h"
#include "../math.h"
#include "denoise/denoise_operator.cuh"
#include "denoise/kdtree.h"
#include "noise_operator.cuh"
#include "rake_operator.cuh"
#include "range_filter_operator.cuh"
#include "rotate_operator.cuh"
#include "sample_filter_operator.cuh"
#include "session_operator_host.h"

#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/containers/device_array.hpp>
#include <pcl/gpu/segmentation/gpu_extract_clusters.h>
#include <pcl/gpu/segmentation/impl/gpu_extract_clusters.hpp>

#include "pcl/test.gen.h"

namespace pc::operators {

	pc::types::PointCloud apply(const pc::types::PointCloud& point_cloud,
		const OperatorList& operator_list) {
		thrust::device_vector<position> positions(point_cloud.size());
		thrust::copy(point_cloud.positions.begin(), point_cloud.positions.end(),
			positions.begin());

		thrust::device_vector<color> colors(point_cloud.size());
		thrust::copy(point_cloud.colors.begin(), point_cloud.colors.end(),
			colors.begin());

		thrust::device_vector<int> indices(point_cloud.size());
		thrust::sequence(indices.begin(), indices.end());

		auto operator_output_begin = thrust::make_zip_iterator(
			thrust::make_tuple(positions.begin(), colors.begin(), indices.begin()));
		auto operator_output_end = thrust::make_zip_iterator(
			thrust::make_tuple(positions.end(), colors.end(), indices.end()));

		for (auto& operator_host_ref : operator_list) {
			operator_output_end = pc::operators::SessionOperatorHost::run_operators(
				operator_output_begin, operator_output_end,
				operator_host_ref.get()._config);
		}

		cudaDeviceSynchronize();

		pc::types::PointCloud result{};

		auto output_point_count =
			std::distance(operator_output_begin, operator_output_end);

		result.positions.resize(output_point_count);
		result.colors.resize(output_point_count);

		thrust::copy(positions.begin(), positions.begin() + output_point_count,
			result.positions.begin());
		thrust::copy(colors.begin(), colors.begin() + output_point_count,
			result.colors.begin());

		return result;
	}

	operator_in_out_t
		SessionOperatorHost::run_operators(operator_in_out_t begin,
			operator_in_out_t end,
			OperatorHostConfiguration& host_config) {

		for (auto& operator_config : host_config.operators) {
			std::visit(
				[&begin, &end](auto&& config) {
					if (!config.enabled) {
						return;
					}

					using T = std::decay_t<decltype(config)>;

					ZoneScopedN(T::Name);

					// Transform operators
					if constexpr (std::is_same_v<T, NoiseOperatorConfiguration>) {
						thrust::transform(begin, end, begin, NoiseOperator{ config });
					}
					else if constexpr (std::is_same_v<T, RotateOperatorConfiguration>) {
						thrust::transform(begin, end, begin, RotateOperator(config));
					}
					else if constexpr (std::is_same_v<T, RakeOperatorConfiguration>) {
						thrust::transform(begin, end, begin, RakeOperator(config));
					}

					// PCL
					else if constexpr (std::is_same_v<T, PCLClusteringConfiguration>) {

						using namespace std::chrono;
						using namespace std::chrono_literals;

						auto start_time = system_clock::now();

						//std::thread([positions = thrust::host_vector<pc::types::position>(

						auto& host = *SessionOperatorHost::instance;

						host.pcl_queue.emplace(pcl_input_frame{
							.timestamp = 0,
							.positions = thrust::host_vector<pc::types::position>(
								thrust::get<0>(begin.get_iterator_tuple()),
								thrust::get<0>(end.get_iterator_tuple()))
						});

						// TODO move these to draw function in session_operator_host.cc

						if (config.draw_voxels) {
							auto latest_voxels = SessionOperatorHost::instance->latest_voxels.load();
							if (latest_voxels != nullptr) {
								// TODO this is too slow
								auto voxel_count = latest_voxels->size();
								for (int i = 0; i < voxel_count; i++) {
									pcl::PointXYZ p = latest_voxels->points[i];
									// mm to metres
									host.set_voxel(i, { p.x / 1000.0f, p.y / 1000.0f, p.z / 1000.0f });
								}
							}
						}

						if (config.draw_clusters) {
							auto latest_clusters_ptr = SessionOperatorHost::instance->latest_clusters.load();
							if (latest_clusters_ptr != nullptr) {
								auto& latest_clusters = *latest_clusters_ptr;
								auto cluster_count = latest_clusters.size();
								pc::logger->debug("Cluster count: {}", cluster_count);
								for (int i = 0; i < cluster_count; i++) {
									pc::AABB aabb = latest_clusters[i];
									auto position = aabb.center() / 1000.0f; // mm to metres
									auto size = aabb.extents() / 1000.0f;
									host.set_cluster(i, position, size);
								}
							}
						}

						//const auto id = config.id;

						//static std::deque<std::future<void>> pcl_tasks;
						//static constexpr size_t max_tasks = 5;

						// static std::atomic<uid> box_id = 0;

						// TODO this thread detach is really bad, need to manage ongoing tasks
						//std::thread([id]() {

						//	std::this_thread::sleep_for(120ms);
						//	box_id.store(id);

						//}).detach();

						//pcl_tasks.push_back(std::async(std::launch::async, [id]() {
						//}));

						//while (pcl_tasks.size() > max_tasks) {
						//	pcl_tasks.pop_front();
						//}

						//if (box_id != 0) {
						//	pc::logger->info("setting box_id: {}", (int)box_id);
						//	 SessionOperatorHost::instance->set_random_box(box_id.load());
						//	box_id = 0;
						//}

						// copy positions from gpu into CPU thread

						// TODO detaching these threads is possibly dangerous
						// need to place them inside a collection of std::future or something but 
						// it can't be blocking to remove the futures from that collection

						//struct VoxelData {
						//	pcl::PointXYZ position;
						//	float size;
						//};

						//static std::mutex voxel_access;
						//static std::optional<std::vector<VoxelData>> voxel_results;

						//std::thread([positions = thrust::host_vector<pc::types::position>(
						//	thrust::get<0>(begin.get_iterator_tuple()),
						//	thrust::get<0>(end.get_iterator_tuple()))]() {

						//	//  convert PointCloud.positions into PCL structures
						//	auto pcl_positions = thrust::host_vector<pcl::PointXYZ>(positions.size());
						//	thrust::transform(positions.begin(), positions.end(), pcl_positions.begin(),
						//		PositionToPointXYZ());

						//	// and copy the transformed result into a PCL PointCloud object
						//	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
						//	cloud->points.resize(positions.size());
						//	std::copy(pcl_positions.begin(), pcl_positions.end(), cloud->points.begin());

						//	pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

						//	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
						//	voxel_grid.setInputCloud(cloud);
						//	voxel_grid.setLeafSize(100.0f, 100.0f, 100.0f); // 100mm ??

						//	voxel_grid.filter(*filtered_cloud);

						//	//// dump into current_voxels
						//	//std::vector<VoxelData> voxels;
						//	//for (const auto& point : filtered_cloud->points) {
						//	//	VoxelData voxel{
						//	//		.position = point,
						//	//		.size = 0.01f
						//	//	};
						//	//	voxels.push_back(voxel);
						//	//}

						//	//// Update the current_voxels
						//	//// std::lock_guard lock(voxel_access);
						//	//voxel_results = std::move(voxels);

						//}).detach();

						// if current voxels have a value, read them then set current_voxels to no value again
						// {
							// std::lock_guard lock(voxel_access);
							//if (voxel_results.has_value()) {
							//	pc::logger->debug(voxel_results->size());
							//	voxel_results.reset();
							//}
						// }

						//// convert result back to position
						//auto result_count = filtered_cloud->width * filtered_cloud->height;
						//std::transform(filtered_cloud->points.begin(), filtered_cloud->points.end(),
						//	thrust::get<0>(begin.get_iterator_tuple()), PointXYZToPosition());

						// -------------

						// end = begin + result_count;

						// now that the pcl position structures are in place in device memory, wrap them 
						// in a PCL collection & point cloud type so they can be used in PCL algorithms
						//auto pcl_positions_ptr = thrust::raw_pointer_cast(pcl_positions.data());
						//auto gpu_cloud = pcl::gpu::Octree::PointCloud(pcl_positions_ptr, pcl_positions.size());

						// build the octree
						//pcl::gpu::Octree::Ptr octree(new pcl::gpu::Octree);
						//octree->setCloud(gpu_cloud);
						//octree->build();

						//// Voxel grid dimensions
						//int grid_size_x = 10, grid_size_y = 10, grid_size_z = 10;
						//int total_voxels = grid_size_x * grid_size_y * grid_size_z;
						//// Min and max values for x, y, and z axes
						//float min_x = -1000.0f, max_x = 1000.0f;
						//float min_y = -1000.0f, max_y = 1000.0f;
						//float min_z = -1000.0f, max_z = 1000.0f;
						//// voxel size diagonal is needed for octree radius-based search
						//float voxel_size_x = (max_x - min_x) / grid_size_x;
						//float voxel_size_y = (max_y - min_y) / grid_size_y;
						//float voxel_size_z = (max_z - min_z) / grid_size_z;
						//float voxel_diagonal = std::sqrt(voxel_size_x * voxel_size_x +
						//	voxel_size_y * voxel_size_y + voxel_size_z * voxel_size_z);
						//float search_radius = voxel_diagonal * 1.1f;

						// thrust::device_vector<int> voxel_indices(total_voxels);
						//thrust::host_vector<int> voxel_indices_host(total_voxels);
						//thrust::sequence(voxel_indices_host.begin(), voxel_indices_host.end());
						//for (int i = 0; i < voxel_indices_host.size(); i++) {
						//	auto p = voxel_indices_host[i];
						//	pc::logger->debug("{}: {}", i, p);
						//}

						// Create a device vector to hold the voxel grid points
						// thrust::device_vector<pcl::PointXYZ> voxel_grid(total_voxels);
						//thrust::host_vector<pcl::PointXYZ> host_grid(total_voxels);
						// fill the voxel list with evenly distributed positions
						//thrust::transform(voxel_indices_host.begin(), voxel_indices_host.end(),
						//	host_grid.begin(), GenerateVoxelGrid());
						// thrust::copy(voxel_grid.begin(), voxel_grid.end(), host_grid.begin());
						//for (int i = 0; i < host_grid.size(); i++) {
						//	auto p = host_grid[i];
						//	pc::logger->debug("{}: ({}, {}, {})", i, p.x, p.y, p.z);
						//}

						//// wrap back into PCL type
						//auto voxel_grid_ptr = thrust::raw_pointer_cast(voxel_grid.data());
						//pcl::gpu::Octree::Queries voxel_grid_queries(voxel_grid_ptr, total_voxels);

						//// For each point in the voxel grid, check for neighbors up to max
						//constexpr int neighbor_threshold = 3;
						//pcl::gpu::NeighborIndices search_results(total_voxels, neighbor_threshold);
						//octree->radiusSearch(voxel_grid_queries, search_radius, neighbor_threshold, search_results);

						//// and filter matching points with enough neighbors 
						//thrust::device_vector<int> filtered_voxel_indices(total_voxels);

						//auto end_it = thrust::copy_if(
						//	thrust::make_counting_iterator(0),
						//	thrust::make_counting_iterator(total_voxels),
						//	filtered_voxel_indices.begin(),
						//	NeighborThresholdFilter{neighbor_threshold, search_results.sizes.ptr()}
						//);
						//auto filtered_voxel_count = thrust::distance(filtered_voxel_indices.begin(), end_it);
						//filtered_voxel_indices.resize(filtered_voxel_count);

						//pc::logger->info("f: {}", filtered_voxel_count);

						//auto indexed_voxels_begin = thrust::make_zip_iterator(
						//	thrust::make_tuple(voxel_grid.begin(), thrust::make_counting_iterator(0)));
						//auto indexed_voxels_end = thrust::make_zip_iterator(
						//	thrust::make_tuple(voxel_grid.end(), thrust::make_counting_iterator(total_voxels)));

						//thrust::transform(indexed_voxels_begin, indexed_voxels_end, begin,
						//	VoxelToPointcloud(thrust::raw_pointer_cast(filtered_voxel_indices.data()), filtered_voxel_count));

						//end = begin + filtered_voxel_count;

						auto end_time = system_clock::now();
						auto duration_us = duration_cast<microseconds>(end_time - start_time);
						auto duration_ms = duration_us.count() / 1000.0f;

						pc::logger->debug("pcl main took {:.2f}ms", duration_ms);

					}

					// Filters
					else if constexpr (std::is_same_v<T, SampleFilterOperatorConfiguration>) {
						end = thrust::copy_if(begin, end, begin, SampleFilterOperator(config));
					}
					else if constexpr (std::is_same_v<T, RangeFilterOperatorConfiguration>) {

						// TODO: this is RangeFilter Operator behaviour,
						// probs needs to be inside that class somehow

							// the operator host should really not contain logic relating to
					  // individual operators, it should simply call them

					  // perhaps we completely replace the thrust::copy_if with an
					  // operator calling function that returns a new end if the config
					  // requires it

						auto starting_point_count = thrust::distance(begin, end);

						thrust::device_vector<position> filtered_positions(
							starting_point_count);
						thrust::device_vector<color> filtered_colors(starting_point_count);
						thrust::device_vector<int> filtered_indices(starting_point_count);

						auto filtered_begin = thrust::make_zip_iterator(thrust::make_tuple(
							filtered_positions.begin(), filtered_colors.begin(),
							filtered_indices.begin()));
						auto filtered_end = thrust::copy_if(begin, end, filtered_begin,
							RangeFilterOperator{ config });

						int fill_count = thrust::distance(filtered_begin, filtered_end);
						config.fill.fill_count = fill_count;

						if (!config.bypass) {
							end = thrust::copy(filtered_begin, filtered_end, begin);
						}

						if (fill_count > config.fill.count_threshold) {

							const auto fill_value =
								fill_count / static_cast<float>(config.fill.max_fill);
							const auto fill_proportion =
								fill_count / static_cast<float>(starting_point_count);

							config.fill.fill_value = fill_value;
							config.fill.proportion = fill_proportion;

							auto minmax_x_points = thrust::minmax_element(
								filtered_begin, filtered_end, MinMaxXComparator{});
							auto minmax_y_points = thrust::minmax_element(
								filtered_begin, filtered_end, MinMaxYComparator{});
							auto minmax_z_points = thrust::minmax_element(
								filtered_begin, filtered_end, MinMaxZComparator{});

							const auto& min_x =
								thrust::get<0>(*minmax_x_points.first).operator position().x /
								1000.0f;
							const auto& max_x = thrust::get<0>(*minmax_x_points.second)
								.
								operator position()
								.x /
								1000.0f;
							const auto& min_y =
								thrust::get<0>(*minmax_y_points.first).operator position().y /
								1000.0f;
							const auto& max_y = thrust::get<0>(*minmax_y_points.second)
								.
								operator position()
								.y /
								1000.0f;
							const auto& min_z =
								thrust::get<0>(*minmax_z_points.first).operator position().z /
								1000.0f;
							const auto& max_z = thrust::get<0>(*minmax_z_points.second)
								.
								operator position()
								.z /
								1000.0f;

							const float box_min_x = config.position.x - config.size.x;
							const float box_max_x = config.position.x + config.size.x;
							const float box_min_y = config.position.y - config.size.y;
							const float box_max_y = config.position.y + config.size.y;
							const float box_min_z = config.position.z - config.size.z;
							const float box_max_z = config.position.z + config.size.z;

							const float min_x_value = pc::math::remap(
								box_min_x, box_max_x, 0.0f, 1.0f, min_x, true);
							const float max_x_value = pc::math::remap(
								box_min_x, box_max_x, 0.0f, 1.0f, max_x, true);
							const float min_y_value = pc::math::remap(
								box_min_y, box_max_y, 0.0f, 1.0f, min_y, true);
							const float max_y_value = pc::math::remap(
								box_min_y, box_max_y, 0.0f, 1.0f, max_y, true);
							const float min_z_value = pc::math::remap(
								box_min_z, box_max_z, 0.0f, 1.0f, min_z, true);
							const float max_z_value = pc::math::remap(
								box_min_z, box_max_z, 0.0f, 1.0f, max_z, true);

							config.minmax.min_x = min_x_value;
							config.minmax.max_x = max_x_value;
							config.minmax.min_y = min_y_value;
							config.minmax.max_y = max_y_value;
							config.minmax.min_z = min_z_value;
							config.minmax.max_z = max_z_value;

							// for a very basic 'energy' parameter, we can calculate the
							// volume and then track changes in the volume over time
							constexpr static auto calculate_volume = [](const auto& mm) {
								return (mm.max_x - mm.min_x) * (mm.max_y - mm.min_y) *
									(mm.max_z - mm.min_z);
								};

							// // and we keep a cache of the last volume per range operator
							// // instance to be able to calculate the volume change over time
							// using uid = unsigned long int;
							// static std::unordered_map<uid, std::tuple<float, float>>
							// 	  volume_cache; // returns the last time and last volume value

								// using namespace std::chrono;
							// static const auto start_time = duration_cast<milliseconds>(
							// 	  high_resolution_clock::now().time_since_epoch());
							// const auto now = duration_cast<milliseconds>(
							// 	  high_resolution_clock::now().time_since_epoch());
							// const float elapsed_seconds =
							// 	  (now.count() - start_time.count()) / 1000.0f;

							// if (!volume_cache.contains(config.id)) {
							// 	volume_cache[config.id] = {elapsed_seconds, 0.0f};
							// }

							// const auto [cache_time, last_volume] = volume_cache[config.id];
							// if (elapsed_seconds - cache_time >=
							// 	  config.minmax.volume_change_timespan) {
							// 	const auto volume = calculate_volume(config.minmax);
							// 	config.minmax.volume_change = std::abs(volume - last_volume);
								//   volume_cache[config.id] = {elapsed_seconds, volume};
								// }

						}
						else {
							config.fill.fill_value = 0;
							config.fill.proportion = 0;
							config.minmax.min_x = 0;
							config.minmax.max_x = 0;
							config.minmax.min_y = 0;
							config.minmax.max_y = 0;
							config.minmax.min_z = 0;
							config.minmax.max_z = 0;
						}

						// if (config.fill.publish) {
						// publisher::publish_all(
						//     "fill_value", std::array<float, 1>{config.fill.fill_value},
						//     {"operator", "range_filter", std::to_string(config.id),
						//      "fill"});
						// publisher::publish_all(
						//     "proportion", std::array<float, 1>{config.fill.proportion},
						//     {"operator", "range_filter", std::to_string(config.id),
						//      "fill"});
						// }

						// TODO these three kernels could probably be fused into one
						// if (config.minmax.publish) {
						//   auto &mm = config.minmax;
						//   auto id = std::to_string(config.id);
						//   publisher::publish_all(
						// 			     "min_x", std::array<float, 1>{mm.min_x},
						// 			     {"operator", "range_filter", id,
						// "minmax"});
						//   publisher::publish_all(
						// 			     "max_x", std::array<float, 1>{mm.max_x},
						// 			     {"operator", "range_filter", id,
						// "minmax"});
						//   publisher::publish_all(
						// 			     "min_y", std::array<float, 1>{mm.min_y},
						// 			     {"operator", "range_filter", id,
						// "minmax"});
						//   publisher::publish_all(
						// 			     "max_y", std::array<float, 1>{mm.max_y},
						// 			     {"operator", "range_filter", id,
						// "minmax"});
						//   publisher::publish_all(
						// 			     "min_z", std::array<float, 1>{mm.min_z},
						// 			     {"operator", "range_filter", id,
						// "minmax"});
						//   publisher::publish_all(
						// 			     "max_z", std::array<float, 1>{mm.max_z},
						// 			     {"operator", "range_filter", id,
						// "minmax"});
						// }
					}
				},
				operator_config);
		}

		return end;
	};

	operator_in_out_t apply(operator_in_out_t begin, operator_in_out_t end,
		const OperatorList& operator_list) {
		for (auto& operator_host_ref : operator_list) {
			auto& operator_host = operator_host_ref.get();
			end = pc::operators::SessionOperatorHost::run_operators(
				begin, end, operator_host._config);
		}
		return end;
	}

} // namespace pc::operators
