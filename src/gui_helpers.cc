#include "gui_helpers.h"
#include <spdlog/spdlog.h>

namespace pc::gui {

unsigned int _parameter_index;

void begin_gui_helpers() { _parameter_index = 0; }

bool begin_tree_node(std::string_view name, bool &open) {
  ImGui::PushID(_parameter_index++);
  auto node_flags = ImGuiTreeNodeFlags_None;
  if (open) node_flags = ImGuiTreeNodeFlags_DefaultOpen;
  open = ImGui::TreeNodeEx(name.data(), node_flags);
  ImGui::PopID();
  return open;
}
} // namespace pc::gui
