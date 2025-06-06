#include "snapshots.h"
#include <Corrade/Utility/Debug.h>
#include <chrono>
#include <imgui.h>
#include <numeric>
#include "logger.h"

// TODO capturing a synthesized pointcloud shouldn't be coming from the
// sensors namespace
#include "devices/device.h"
#include "operators/operator_host_config.gen.h"

namespace pc::snapshots {

using namespace pc::types;

std::vector<PointCloud> frames;

PointCloud point_cloud(){
  return std::reduce(
      frames.begin(), frames.end(), PointCloud{},
      [](auto a, auto b) -> PointCloud { return a + b; });
};


void Snapshots::draw_imgui_window() {
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
  pc::logger->info("Capturing frame unimplemented");
  // TODO
  // frames.push_back({pc::devices::synthesized_point_cloud(empty_list)});
}

} // namespace pc::snapshots
