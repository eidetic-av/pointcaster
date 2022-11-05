#include "snapshots.h"
#include <Corrade/Utility/Debug.h>
#include <chrono>
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <numeric>

// TODO capturing a synthesized pointcloud shouldn't be coming from the
// sensors namespace
#include "devices/device.h"

namespace bob::pointcaster::snapshots {

using namespace bob::types;

std::vector<PointCloud> frames;

PointCloud pointCloud(){
  return std::reduce(
      frames.begin(), frames.end(), PointCloud{},
      [](auto a, auto b) -> PointCloud { return a + b; });
};


void Snapshots::drawImGuiWindow() {
  using namespace ImGui;
  PushID("Snapshots");
  SetNextWindowPos({350.0f, 500.0f}, ImGuiCond_FirstUseEver);
  SetNextWindowSize({200.0f, 100.0f}, ImGuiCond_FirstUseEver);
  SetNextWindowBgAlpha(0.8f);

  Begin("Snapshots", nullptr);
  PushItemWidth(GetWindowWidth() * 0.8f);

  bool clear_frames;
  if (Checkbox("Clear Frames", &clear_frames)) frames.clear();

  BeginTable("Table", 3);
  TableSetupColumn("Frame");
  TableSetupColumn("Capture");
  TableSetupColumn("Sequence");
  TableHeadersRow();
  TableNextRow();
  TableNextColumn();
  Text("Frame title");
  TableNextColumn();
  bool do_capture;
  if (Checkbox("##Capture", &do_capture)) capture();
  EndTable();

  // Checkbox("Enable compression", nullptr);

  PopItemWidth();
  End();
  PopID();
}

void Snapshots::capture() {
  spdlog::info("Capturing frame");
  frames.push_back({bob::sensors::synthesizedPointCloud()});
}

} // namespace bob::pointcaster
