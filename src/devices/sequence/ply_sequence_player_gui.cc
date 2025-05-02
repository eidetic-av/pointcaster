#include "../../fonts/IconsFontAwesome6.h"
#include "../../gui/catpuccin.h"
#include "../../gui/widgets.h"
#include "ply_sequence_player.h"
#include <imgui.h>

namespace pc::devices {

bool PlySequencePlayer::draw_controls() {
  using namespace ImGui;
  bool updated = false;
  auto& config = this->config();

  static const float controls_height = GetTextLineHeightWithSpacing() * 2;
  ImVec2 controls_size{GetContentRegionAvail().x, controls_height};

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {5, 5});
  if (ImGui::BeginChild(
          std::format("##controls/{}", pc::gui::_parameter_index++).c_str(),
          controls_size,
          ImGuiChildFlags_Border | ImGuiWindowFlags_NoScrollbar)) {}

  using namespace catpuccin::imgui;
  using catpuccin::with_alpha;

  if (gui::draw_icon_button(ICON_FA_PLAY, false, with_alpha(mocha_blue, 0.75f),
                            mocha_blue)) {
    if (!config.playing) {
      config.playing = true;
      updated = true;
    }
  };

  ImGui::SameLine();
  ImGui::Dummy({5, 0});
  ImGui::SameLine();

  if (gui::draw_icon_button(ICON_FA_PAUSE, false,
                            with_alpha(mocha_surface2, 0.75f),
                            mocha_surface2)) {
    if (config.playing) {
      config.playing = false;
      updated = true;
    }
  };

  ImGui::SameLine();
  ImGui::Dummy({15, 0});
  ImGui::SameLine();

  if (gui::draw_icon_button(ICON_FA_STOP, false,
                            with_alpha(mocha_maroon, 0.75f), mocha_maroon)) {
    if (config.playing) {
      config.playing = false;
      updated = true;
    }
    if (config.current_frame != 0) {
      config.current_frame = 0;
      updated = true;
    }
  };

  ImGui::SameLine();
  ImGui::Dummy({5, 0});
  ImGui::SameLine();

  if (gui::draw_icon_button(ICON_FA_BACKWARD_STEP, false,
                            with_alpha(mocha_maroon, 0.75f), mocha_maroon)) {
    if (config.current_frame != 0) {
      config.current_frame = 0;
      updated = true;
    }
  };

  ImGui::EndChild();
  ImGui::PopStyleVar();

  return updated;
}

} // namespace pc::devices