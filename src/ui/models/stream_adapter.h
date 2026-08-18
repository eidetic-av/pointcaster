#pragma once

#include <optional>
#include <pointcaster/point_cloud.h>

// what a scene object reads to draw an operator's output stream.
// only one of these returns a value: whichever type the operator writes
class StreamAdapter {
public:
  virtual ~StreamAdapter() = default;

  virtual pc::AabbListPtr aabb_list() { return nullptr; }
  virtual pc::VoxelisedCloudPtr voxelised_cloud() { return nullptr; }
  virtual std::optional<pc::position_bounds> bounds() { return std::nullopt; }
  virtual std::optional<pc::radius> radius() { return std::nullopt; }
};
