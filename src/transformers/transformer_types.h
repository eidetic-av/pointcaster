#pragma once

#include "../structs.h"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>

namespace pc::transformers {

  typedef thrust::tuple<pc::types::position, pc::types::color> point_t;

  typedef thrust::zip_iterator<
      thrust::tuple<thrust::device_vector<pc::types::position>::iterator,
                    thrust::device_vector<pc::types::color>::iterator>>
      transformer_in_out_t;
}
