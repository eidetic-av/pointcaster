#pragma once

#include <memory>
#include <pointcaster/point_cloud.h>

class PointCloudAdapter {
public:
  virtual std::shared_ptr<pc::PointCloud> point_cloud() = 0;

  virtual std::shared_ptr<std::vector<std::byte>> render_data() {
    return nullptr;
  }

};