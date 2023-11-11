#pragma once

#include "../structs.h"
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace pc::operators {

typedef thrust::tuple<pc::types::position, pc::types::color> point_t;
typedef thrust::tuple<pc::types::position, pc::types::color, int> indexed_point_t;

typedef thrust::zip_iterator<
    thrust::tuple<thrust::device_vector<pc::types::position>::iterator,
		  thrust::device_vector<pc::types::color>::iterator,
		  thrust::device_vector<int>::iterator>>
    operator_in_out_t;
} // namespace pc::operators
