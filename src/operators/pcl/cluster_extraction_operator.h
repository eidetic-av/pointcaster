#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/parallel_pipeline.h>
#include <thrust/host_vector.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "../../aabb.h"
#include "../../structs.h"

namespace pc::operators::pcl_cpu {

using uid = unsigned long;

struct ClusterExtractionConfiguration {
	uid id;
	bool enabled = true;
	bool draw_voxels = false;
	bool draw_clusters = true;
	int voxel_leaf_size = 200; // @minmax(100, 1000)
	bool publish_voxels = false;
	int cluster_tolerance = 270; // @minmax(120, 1200)
	int cluster_voxel_count_min = 10; // @minmax(1, 300)
	int cluster_voxel_count_max = 100; // @minmax(1, 300)
	int cluster_size_min = 10; // @minmax(0, 5000)
	int cluster_size_max = 1000; // @minmax(0, 5000)
	bool publish_clusters = false;
};

	class ClusterExtractionPipeline {

	public:
		struct InputFrame {
			unsigned long timestamp;
			ClusterExtractionConfiguration extraction_config;
			thrust::host_vector<pc::types::position> positions;
		};

		static ClusterExtractionPipeline& instance();

		oneapi::tbb::concurrent_bounded_queue<InputFrame> input_queue;
		std::atomic<pcl::PointCloud<pcl::PointXYZ>::Ptr> current_voxels;
		std::atomic<std::shared_ptr<std::vector<pc::AABB>>> current_clusters;

		ClusterExtractionPipeline();

		ClusterExtractionPipeline(const ClusterExtractionPipeline&) = delete;
		ClusterExtractionPipeline& operator=(const ClusterExtractionPipeline&) = delete;
		ClusterExtractionPipeline(ClusterExtractionPipeline&&) = delete;
		ClusterExtractionPipeline& operator=(ClusterExtractionPipeline&&) = delete;

	private:
		struct IngestTask {
			oneapi::tbb::concurrent_bounded_queue<InputFrame>& input_queue;
			bool& host_shutdown;
			InputFrame* operator()(tbb::flow_control& fc) const;
		};

		struct ExtractTask {
			std::atomic<pcl::PointCloud<pcl::PointXYZ>::Ptr>& current_voxels;
			std::atomic<std::shared_ptr<std::vector<pc::AABB>>>& current_clusters;
			void operator()(InputFrame* frame) const;
		};
		std::jthread _host_thread;
		bool _host_shutdown = false;
	};

	// Util for converting to pcl types
	struct PositionToPointXYZ {
		__host__ __device__
			pcl::PointXYZ operator()(const pc::types::position& pos) const {
			pcl::PointXYZ point;
			point.x = static_cast<float>(pos.x);
			point.y = static_cast<float>(pos.y);
			point.z = static_cast<float>(pos.z);
			return point;
		}
	};

	// Util for converting from pcl types
	struct PointXYZToPosition {
		__host__ __device__
			pc::types::position operator()(const pcl::PointXYZ& pos) const {
			return pc::types::position{ static_cast<short>(pos.x), static_cast<short>(pos.y), static_cast<short>(pos.z) };
		}
	};

}
