#pragma once
#include <memory>
#include <thread>
#include <vector>
#include <pointclouds.h>
#include "structs.h"

namespace pc::snapshots {

extern std::vector<pc::types::PointCloud> frames;
extern pc::types::PointCloud pointCloud();

struct SnapshotsConfiguration {
};

class Snapshots {
public:
  Snapshots(){};
  Snapshots(SnapshotsConfiguration config) : _config{config} {}
  void draw_imgui_window();
private:
  SnapshotsConfiguration _config{};
  void capture();
};
} // namespace pc::snapshots
