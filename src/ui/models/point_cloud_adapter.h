#pragma once

#include <pointcaster/point_cloud.h>

class PointCloudAdapter {
public:
  virtual const pc::PointCloud &point_cloud() = 0;
};