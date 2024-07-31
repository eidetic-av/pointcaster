#pragma once
#include <pcl/point_types.h>
#include "../../identifiable.h"
#include "../operator.h"

namespace pc::operators {

using uid = unsigned long int;

//struct AABBox {
//	pc::types::Float3 min;
//	pc::types::Float3 max;
//};
//
//using EuclidianCluster = AABBox;
//
//static inline std::atomic<std::shared_ptr<std::vector<EuclidianCluster>>>
//	euclidian_clusters(std::make_shared<std::vector<EuclidianCluster>>());

struct GenerateVoxelGrid : public thrust::unary_function<int, pcl::PointXYZ> {
	__host__ __device__
		pcl::PointXYZ operator()(int index) const {
		return pcl::PointXYZ(static_cast<float>(index), static_cast<float>(index), static_cast<float>(index));
	}
};

struct NeighborThresholdFilter {
	int neighbor_threshold;
	const int* neighbor_counts;

    __host__ __device__ 
	bool operator()(int idx) const {
        return neighbor_counts[idx] >= neighbor_threshold;
    }
};

typedef thrust::tuple<pcl::PointXYZ, int> indexed_voxel_t;

struct VoxelToPointcloud : thrust::unary_function<indexed_voxel_t, indexed_point_t> {
	const int* _filtered_indices;
	int _filtered_count;

	VoxelToPointcloud(const int* filtered_indices, int filtered_count) :
		_filtered_indices(filtered_indices), _filtered_count(filtered_count) { };

	__device__ indexed_point_t operator()(indexed_voxel_t voxel) const;
};


} // namespace pc::operators
