#pragma once

namespace pc::gui {

struct IRightClickMenu {
  virtual ~IRightClickMenu() = default;
  virtual bool draw_right_click_menu() = 0;
};

} // namespace pc::gui
