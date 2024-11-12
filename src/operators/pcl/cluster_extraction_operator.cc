#include <oneapi/tbb.h>
#include <memory>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include "cluster_extraction_operator.gen.h"
#include "../../logger.h"
#include "../../publisher/publisher.h"

namespace pc::operators::pcl_cpu {

	ClusterExtractionPipeline& ClusterExtractionPipeline::instance() {
		static ClusterExtractionPipeline instance;
		return instance;
	}

	ClusterExtractionPipeline::ClusterExtractionPipeline()
	{
		auto max_concurrency = tbb::info::default_concurrency();
		input_queue.set_capacity(max_concurrency);

		pc::logger->info("Starting Cluster Extraction TBB Pipeline (max_concurrency = {})",
			max_concurrency);

		_host_thread = std::jthread([this, max_concurrency](std::stop_token st) {
if (st.stop_requested()) return;
			tbb::parallel_pipeline(
				max_concurrency,
				tbb::make_filter<void, InputFrame*>(
					tbb::filter_mode::serial_in_order, IngestTask{ input_queue, st }) &
				tbb::make_filter<InputFrame*, void>(
					tbb::filter_mode::parallel, ExtractTask{ current_voxels, current_clusters })
			);
		});
	}

	ClusterExtractionPipeline::~ClusterExtractionPipeline()
	{
		_host_thread.request_stop();
	}

	// TODO swap raw ptrs for shared pointers in pipeline stages

	ClusterExtractionPipeline::InputFrame* ClusterExtractionPipeline::IngestTask::operator()(tbb::flow_control& fc) const {
		using namespace std::chrono_literals;
		auto* new_frame = new InputFrame;
		while (!input_queue.try_pop(*new_frame)) {
			if (st.stop_requested()) {
				delete new_frame;
				fc.stop();
				return nullptr;
			}
			std::this_thread::sleep_for(2ms);
		}
		return new_frame;
	}


	void ClusterExtractionPipeline::ExtractTask::operator()(ClusterExtractionPipeline::InputFrame* frame) const {
		if (frame) {
			using namespace std::chrono;
			using namespace std::chrono_literals;

			auto start_time = system_clock::now();

			auto& config = frame->extraction_config;

			// TODO check timestamp against current time

			// TODO configuration options for voxel sampling and clustering

			auto& positions = frame->positions;
			auto count = positions.size();
			if (count == 0) return;

			//  convert position type into PCL structures
			auto pcl_positions = thrust::host_vector<pcl::PointXYZ>(count);
			thrust::transform(positions.begin(), positions.end(), pcl_positions.begin(), PositionToPointXYZ());

			// and copy the transformed result into a PCL PointCloud object
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
			cloud->points.resize(count);
			std::copy(pcl_positions.begin(), pcl_positions.end(), cloud->points.begin());

			// filter the cloud into a voxel grid
			pcl::PointCloud<pcl::PointXYZ>::Ptr voxelised_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
			voxel_grid.setInputCloud(cloud);
			voxel_grid.setLeafSize(config.voxel_leaf_size, config.voxel_leaf_size, config.voxel_leaf_size); // mm
			voxel_grid.filter(*voxelised_cloud);

			current_voxels.store(voxelised_cloud);

			if (config.publish_voxels) {
				// interpret the pcl point cloud as standard value types
				static_assert(sizeof(pcl::PointXYZ) == sizeof(std::array<float, 4>));
				auto point_data = reinterpret_cast<std::vector<std::array<float, 4>>&>(voxelised_cloud->points);
				// auto id_string = std::to_string(config.id);
				auto id_string = "pcl";
				publisher::publish_all("voxels", point_data, { id_string });
			}

			auto voxelisation_time = system_clock::now();
			auto voxelisation_us = duration_cast<microseconds>(voxelisation_time - start_time);
			auto voxelisation_ms = voxelisation_us.count() / 1000.0f;
			// pc::logger->debug("voxelisation time: {:.2f}ms", voxelisation_ms);

			// create kdtree
			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
			tree->setInputCloud(voxelised_cloud);

			auto kdgen_time = system_clock::now();
			auto kdgen_us = duration_cast<microseconds>(kdgen_time - voxelisation_time);
			auto kdgen_ms = kdgen_us.count() / 1000.0f;
			// pc::logger->debug("kdgen time: {:.2f}ms", kdgen_ms);

			// perform clustering
			std::vector<pcl::PointIndices> cluster_indices;
			pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
			ec.setClusterTolerance(config.cluster_tolerance); // mm
			ec.setMinClusterSize(config.cluster_voxel_count_min);
			ec.setMaxClusterSize(config.cluster_voxel_count_max);
			ec.setSearchMethod(tree);
			ec.setInputCloud(voxelised_cloud);
			ec.extract(cluster_indices);

			auto cluster_time = system_clock::now();
			auto cluster_us = duration_cast<microseconds>(cluster_time - kdgen_time);
			auto cluster_ms = cluster_us.count() / 1000.0f;
			// pc::logger->debug("cluster time: {:.2f}ms", cluster_ms);

			// calculate bounding boxes
			tbb::concurrent_vector<pc::AABB> cluster_bounds;
			tbb::parallel_for(tbb::blocked_range<size_t>(0, cluster_indices.size()),
				[&cluster_indices, &voxelised_cloud, &cluster_bounds](const tbb::blocked_range<size_t>& r) {
					for (size_t i = r.begin(); i != r.end(); ++i) {
						const auto& indices = cluster_indices[i];

						pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
						for (const auto& index : indices.indices) {
							cloud_cluster->points.push_back(voxelised_cloud->points[index]);
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
			// pc::logger->debug("aabb time: {:.2f}ms", aabb_ms);

			// copy data to host instance
			auto output_bounds = std::make_shared<std::vector<pc::AABB>>(cluster_bounds.size());
			std::copy(cluster_bounds.begin(), cluster_bounds.end(), output_bounds->begin());

			current_clusters.store(output_bounds);

			if (config.publish_clusters) {
				using AABBMinMax = std::array<std::array<float, 3>, 2>;
				auto cluster_count = output_bounds->size();
				std::vector<AABBMinMax> cluster_bounds_data(cluster_count);
				for (int i = 0; i < cluster_count; i++) {
					auto& bounds = output_bounds->at(i);
					cluster_bounds_data[i] = { {
						{ bounds.min.x, bounds.min.y, bounds.min.z },
						{ bounds.max.x, bounds.max.y, bounds.max.z }
					} };
				}
				// auto id_string = std::to_string(config.id);
				auto id_string = "pcl";
				publisher::publish_all("clusters", cluster_bounds_data, { id_string });
			}

			delete frame;
		}
	}
}