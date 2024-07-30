#pragma once

#include "../../structs.h"
#include "test.gen.h"
#include <thrust/functional.h>

namespace pc::operators {

	__device__ indexed_point_t VoxelToPointcloud::operator()(indexed_voxel_t voxel) const {
		auto voxel_index = thrust::get<1>(voxel);
		bool contained = false;
		for (size_t i = 0; i < _filtered_count; ++i) {
			if (voxel_index == _filtered_indices[i]) {
				contained = true;
				break;
			}
		}
		if (contained) {
			auto pos_f = thrust::get<0>(voxel);
			pc::types::position pos_out = { (short)__float2int_rd(pos_f.x),
						(short)__float2int_rd(pos_f.y),
						(short)__float2int_rd(pos_f.z), 0 };
			pc::types::color col{ 255, 0, 0, 255 };
			return thrust::make_tuple(pos_out, col, voxel_index);
		}
		else {
			return thrust::make_tuple(pc::types::position{ 0, 0, 0 }, pc::types::color{ 0, 0, 0, 0 }, voxel_index);
		}
	}

} // namespace pc::operators
