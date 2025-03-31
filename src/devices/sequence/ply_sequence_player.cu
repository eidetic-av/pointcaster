#include "ply_sequence_player.h"

#include "../../operators/operator.h"
#include "../../structs.h"
#include "../transform_filters.cuh"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>
#include <tracy/Tracy.hpp>



namespace pc::devices {

pc::types::PointCloud
PlySequencePlayer::point_cloud(pc::operators::OperatorList operators) {

  if (!config().active) return {};

//   if (!_device_memory_ready) return {};

  ZoneScopedN("PlySequencePlayer::point_cloud");

  const auto &transform = config().transform;
  auto frame = current_frame();

  pc::types::PointCloud cloud =
      _pointcloud_buffer.at(std::min(frame, _pointcloud_buffer.size()));

//   const auto point_indices =
//       std::views::iota(0, static_cast<int>(cloud.size()));
//   std::for_each(std::execution::par, point_indices.begin(), point_indices.end(),
//                 [&](int i) {
//                   // transform positions as per config
//                   const auto &pos = cloud.positions[i];
//                   cloud.positions[i] = {
//                       static_cast<short>(pos.x + transform.translate.x),
//                       static_cast<short>(pos.y + transform.translate.y),
//                       static_cast<short>(pos.z + transform.translate.z)};
//                 });

  return cloud;
}


} // namespace pc::devices