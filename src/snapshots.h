#pragma once
#include <memory>
#include <thread>
#include <vector>
#include <pointclouds.h>

namespace bob::pointcaster::snapshots {

extern std::vector<bob::types::PointCloud> frames;
extern bob::types::PointCloud pointCloud();

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
} // namespace bob::pointcaster
