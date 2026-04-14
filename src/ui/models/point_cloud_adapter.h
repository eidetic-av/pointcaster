#pragma once

#include <memory>
#include <pointcaster/point_cloud.h>

class PointCloudAdapter {
public:
  virtual std::shared_ptr<pc::PointCloud> point_cloud() = 0;
};