#include "main_viewport.h"
#include "widgets.h"
#include "windows.h"

namespace pc::gui {

void draw_main_viewport(PointCaster &app) {

  static std::optional<std::string> remove_session_id;
  static ImVec2 remove_session_pos;
  static std::string remove_session_window_id;
  static std::optional<int> rename_session_index;
  static std::optional<int> new_session_index;
  static ImVec2 rename_tab_pos;
  static std::string rename_edit_buffer;

  constexpr float tab_bar_height = 20;
  static ImVec2 tab_bar_padding = {10.0f, 5.0f};
  ImGuiViewport *vp = ImGui::GetMainViewport();

  auto& workspace = app.workspace;

  // draw the tab bar at the top which can't be docked into and isn't part of
  // the workspace
  ImGui::SetNextWindowPos({vp->Pos.x, vp->Pos.y});
  ImGui::SetNextWindowSize({vp->Size.x, tab_bar_height});
  ImGui::SetNextWindowViewport(vp->ID);
  ImGuiWindowFlags tabbar_flags =
      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings |
      ImGuiWindowFlags_NoDocking;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 5});
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, tab_bar_padding);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::Begin("##TabBar", nullptr, tabbar_flags);
  if (ImGui::BeginTabBar("Sessions", ImGuiTabBarFlags_Reorderable)) {

    auto &style = ImGui::GetStyle();
    ImDrawList *dl = ImGui::GetWindowDrawList();

    // app/file menu button at the start of the tab bar
    if (gui::draw_icon_tab_button(ICON_FA_BARS)) { gui::show_app_menu(); }

    // add new session button at the end of the tab bar
    if (gui::draw_icon_tab_button(ICON_FA_CIRCLE_PLUS,
                                  ImGuiTabItemFlags_Trailing, {-7, 0})) {
      ImGui::OpenPopup("AddSessionPopup");
    }

    gui::draw_app_menu(app);

    gui::push_context_menu_styles();
    if (ImGui::BeginPopup("AddSessionPopup",
                          ImGuiWindowFlags_AlwaysAutoResize)) {
      if (ImGui::MenuItem("Add new session to workspace")) {
        new_session_index = workspace.sessions.size();
        auto new_label_num = *new_session_index;
        auto new_label = std::format("session_{}", new_label_num);
        while (std::ranges::any_of(workspace.sessions, [&](auto &session) {
          return session.label == new_label;
        })) {
          new_label = std::format("session_{}", ++new_label_num);
        }
        workspace.sessions.push_back({.id = uuid::word(), .label = new_label});
        app.sync_session_instances();
        workspace.selected_session_index = workspace.sessions.size() - 1;
      }
      if (ImGui::MenuItem("Load session from disk")) {
        pc::logger->info("Load?");
      }
      ImGui::EndPopup();
    }
    gui::pop_context_menu_styles();

    // tab items for each open session

    constexpr auto tab_width = 125;

    for (int i = 0; i < workspace.sessions.size(); i++) {
      auto &session = workspace.sessions[i];
      ImGuiTabItemFlags item_flags = ImGuiTabItemFlags_None;
      if (new_session_index && i == *new_session_index) {
        item_flags |= ImGuiTabItemFlags_SetSelected;
        new_session_index.reset();
      }

      ImGui::SetNextItemWidth(tab_width);
      auto tab_name = std::format("{}###{}", session.label, session.id);
      if (ImGui::BeginTabItem(tab_name.c_str(), nullptr, item_flags)) {
        workspace.selected_session_index = i;
        ImGui::EndTabItem();
      }

      // rename a session on a double click of its tab
      if (ImGui::IsItemHovered() &&
          ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        rename_session_index = i;
        rename_tab_pos = ImGui::GetItemRectMin();
        rename_edit_buffer = session.label;
        ImGui::OpenPopup("RenameSession");
      }
    }

    auto *session_tab_bar = ImGui::GetCurrentTabBar();

    // calculate close button positions and ids for each tab
    struct CloseButton {
      ImVec2 pos;
      ImGuiID tab_id;
    };
    std::vector<CloseButton> close_buttons;
    close_buttons.resize(session_tab_bar->Tabs.size());
    // iterate the tab count minus a few due to the
    // buttons that are also "tabs" in the bar
    for (int i = 0; i < session_tab_bar->Tabs.size() - 2; i++) {
      auto tab_index = i + 1;
      float close_button_x =
          tab_index * (tab_width + tab_bar_padding.x / 2) - 3;
      float close_button_y = tab_bar_padding.y + 1;
      const auto &tab = session_tab_bar->Tabs[tab_index];
      close_buttons[i] = {{close_button_x, close_button_y}, tab.ID};
    }

    // draw close buttons over top of our tab bar
    // and handle setting up removal of session when clicked
    for (const auto &button : close_buttons) {
      // draw a close session button on the right hand side of each tab
      if (button.pos.x == 0) continue;
      ImGui::SetCursorScreenPos(button.pos);
      if (gui::draw_icon_button(
              ICON_FA_XMARK, false,
              catpuccin::with_alpha(catpuccin::imgui::mocha_text, 0.25f),
              catpuccin::with_alpha(catpuccin::imgui::mocha_text, 0.5f))) {

        for (int i = 0; i < workspace.sessions.size(); ++i) {
          const auto &session = workspace.sessions[i];
          auto tab_name = std::format("{}###{}", session.label, session.id);
          if (ImGui::GetID(tab_name.c_str()) == button.tab_id) {
            pc::logger->info("Should remove: {}", session.label);
            remove_session_id = session.id;
            remove_session_pos = {button.pos.x - 30, button.pos.y + 30};
            remove_session_window_id =
                std::format("Delete Session?###{}", uuid::digit());
            ImGui::OpenPopup(remove_session_window_id.c_str());
            break;
          }
        }
      }
    }

    // handle reordering of sessions by dragging tabs around

    if (session_tab_bar->ReorderRequestTabId) {
      int source_index = -1;
      for (int i = 0; i < workspace.sessions.size(); i++) {
        const auto &session = workspace.sessions[i];
        auto tab_name = std::format("{}###{}", session.label, session.id);
        if (ImGui::GetID(tab_name.c_str()) ==
            session_tab_bar->ReorderRequestTabId) {
          source_index = i;
          break;
        }
      }
      int destination_index =
          source_index + session_tab_bar->ReorderRequestOffset;
      if (source_index >= 0 && destination_index >= 0 &&
          destination_index < workspace.sessions.size()) {
        std::swap(workspace.sessions[source_index],
                  workspace.sessions[destination_index]);
      }
    }

    // draw popup windows & modals

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {10, 5});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {10, 5});
    ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, {10, 5});
    ImGui::SetNextWindowPos(remove_session_pos);
    if (ImGui::BeginPopupModal(remove_session_window_id.c_str(), nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize) &&
        remove_session_id) {
      auto target_session_it = std::ranges::find(
          workspace.sessions, remove_session_id, &PointcasterSession::id);
      auto &target_session = *target_session_it;
      ImGui::Dummy({0, 5});
      ImGui::Text("%s",
                  std::format("Really delete '{}' from the current workspace?",
                              target_session.label)
                      .c_str());
      ImGui::Dummy({0, 10});

      if (ImGui::Button("Delete", ImVec2(80, 0))) {
        int removed_index =
            std::distance(workspace.sessions.begin(), target_session_it);
        workspace.sessions.erase(target_session_it);

        if (workspace.sessions.empty()) {
          workspace.selected_session_index = 0;
        } else if (removed_index == workspace.selected_session_index) {
          if (workspace.selected_session_index >=
              int(workspace.sessions.size()))
            workspace.selected_session_index =
                int(workspace.sessions.size()) - 1;
        } else if (removed_index < workspace.selected_session_index) {
          workspace.selected_session_index--;
        }
        remove_session_id.reset();
        rename_session_index.reset();
        new_session_index.reset();
        ImGui::CloseCurrentPopup();
      }

      ImGui::SameLine();

      if (ImGui::Button("Cancel", ImVec2(80, 0))) {
        ImGui::CloseCurrentPopup();
      }
      ImGui::Dummy({0, 5});

      ImGui::EndPopup();
    }

    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();

    ImGui::SetNextWindowPos(rename_tab_pos, ImGuiCond_Appearing);
    ImGui::SetNextWindowSize({tab_width, -FLT_MIN});
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    if (ImGui::BeginPopup("RenameSession", ImGuiWindowFlags_NoDecoration) &&
        rename_session_index) {
      auto &target_session = workspace.sessions.at(*rename_session_index);
      auto rename_input_label =
          fmt::format("##rename_session_{}", *rename_session_index);
      if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
      ImGui::PushItemWidth(tab_width);
      if (ImGui::InputText(rename_input_label.c_str(), &rename_edit_buffer,
                           ImGuiInputTextFlags_AutoSelectAll |
                               ImGuiInputTextFlags_EnterReturnsTrue)) {
        if (std::ranges::any_of(workspace.sessions, [&](auto &session) {
              return session.label == rename_edit_buffer;
            })) {
          // can't name two sessions the same name
          rename_edit_buffer.clear();
        }
        target_session.label = rename_edit_buffer.empty() ? target_session.label
                                                          : rename_edit_buffer;
        session_label_from_id[target_session.id] = target_session.label;
        ImGui::CloseCurrentPopup();
        new_session_index = *rename_session_index;
        rename_session_index.reset();
        gui::refresh_tooltips = true;
      }
      ImGui::EndPopup();
    }
    ImGui::PopStyleVar();

    ImGui::EndTabBar(); // session tab bar
  }
  ImGui::End();
  ImGui::PopStyleVar();
  ImGui::PopStyleVar();
  ImGui::PopStyleVar();

  // main docking workspace below the tab bar
  ImGui::SetNextWindowPos({vp->Pos.x, vp->Pos.y + tab_bar_height});
  ImGui::SetNextWindowSize({vp->Size.x, vp->Size.y - tab_bar_height});
  ImGui::SetNextWindowViewport(vp->ID);
  ImGuiWindowFlags host_flags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
      ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
      ImGuiWindowFlags_NoBringToFrontOnFocus;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
  ImGui::Begin("##DockSpaceHost", nullptr, host_flags);
  ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
  ImGui::DockSpace(dockspace_id, {0, 0},
                   ImGuiDockNodeFlags_PassthruCentralNode |
                       ImGuiDockNodeFlags_NoDockingInCentralNode);
  ImGui::End();
  ImGui::PopStyleVar();

  // draw the central content inside the workspace - the selected session's
  // camera controller output frames
  ImGui::SetNextWindowDockID(dockspace_id, ImGuiCond_Always);
  ImGuiWindowClass wc = {};
  wc.DockNodeFlagsOverrideSet = ImGuiDockNodeFlags_NoTabBar;
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0);
  ImGui::SetNextWindowClass(&wc);
  ImGui::Begin("CentralContent", nullptr,
               ImGuiWindowFlags_NoCollapse |
                   ImGuiWindowFlags_NoBringToFrontOnFocus);
  {
    auto &selected_session =
        workspace.sessions.at(workspace.selected_session_index);
    auto &camera_controller =
        app.camera_controllers.at(workspace.selected_session_index);
    const auto camera_config = camera_controller->config();

    const auto window_size = ImGui::GetContentRegionAvail();

    if (workspace.layout.hide_ui) { ImGui::SetCursorPosY(0); }

    const auto draw_frame_labels =
        [&camera_controller](ImVec2 viewport_offset) {
          ImGui::PushStyleColor(ImGuiCol_Text, {1.0f, 1.0f, 0.0f, 1.0f});
          for (auto label : camera_controller->labels()) {
            auto &pos = label.position;
            auto x = static_cast<float>(viewport_offset.x + pos.x);
            auto y = static_cast<float>(viewport_offset.y + pos.y);
            ImGui::SetCursorPos({x, y});
            ImGui::Text("%s", label.text.data());
          }
          ImGui::PopStyleColor();
        };

    auto rendering = camera_config.rendering;
    auto scale_mode = rendering.scale_mode;
    const Vector2 frame_space{window_size.x, window_size.y};
    camera_controller->viewport_size = frame_space;

    if (scale_mode == (int)ScaleMode::Span) {

      auto image_pos = ImGui::GetCursorPos();
      ImGuiIntegration::image(camera_controller->color_frame(),
                              {frame_space.x(), frame_space.y()});

      auto analysis = camera_config.analysis;

      if (analysis.enabled && (analysis.contours.draw)) {
        ImGui::SetCursorPos(image_pos);
        ImGuiIntegration::image(camera_controller->analysis_frame(),
                                {frame_space.x(), frame_space.y()});
        draw_frame_labels(image_pos);
      }

    } else if (scale_mode == (int)ScaleMode::Letterbox) {

      float width, height, horizontal_offset, vertical_offset;

      auto frame_aspect_ratio =
          rendering.resolution[0] / static_cast<float>(rendering.resolution[1]);
      auto space_aspect_ratio = frame_space.x() / frame_space.y();

      constexpr auto frame_spacing = 5.0f;

      if (frame_aspect_ratio > space_aspect_ratio) {
        width = frame_space.x();
        height = std::round(width / frame_aspect_ratio);
        horizontal_offset = 0.0f;
        vertical_offset =
            std::max(0.0f, (frame_space.y() - height) / 2.0f - frame_spacing);
      } else {
        height = frame_space.y() - 2 * frame_spacing;
        width = std::round(height * frame_aspect_ratio);
        vertical_offset = 0.0f;
        horizontal_offset = std::max(0.0f, (frame_space.x() - width) / 2.0f);
      }

      ImGui::Dummy({horizontal_offset, vertical_offset});
      if (horizontal_offset != 0.0f) ImGui::SameLine();

      constexpr auto border_size = 1.0f;
      ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, border_size);
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {0, 0});

      ImGui::BeginChildFrame(ImGui::GetID("letterboxed"), {width, height});

      auto image_pos = ImGui::GetCursorPos();
      ImGuiIntegration::image(camera_controller->color_frame(),
                              {width, height});

      auto &analysis = camera_controller->config().analysis;

      if (analysis.enabled && (analysis.contours.draw)) {
        ImGui::SetCursorPos(image_pos);
        ImGuiIntegration::image(camera_controller->analysis_frame(),
                                {width, height});
        draw_frame_labels(image_pos);
      }

      ImGui::EndChildFrame();

      ImGui::PopStyleVar();
      ImGui::PopStyleVar();

      if (horizontal_offset != 0.0f) ImGui::SameLine();
      ImGui::Dummy({horizontal_offset, vertical_offset});
    }

    if (!workspace.layout.hide_ui) {
      // draw_viewport_controls(*camera_controller);
      if (workspace.layout.show_log) { app.draw_onscreen_log(); }
    }

    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_RootWindow)) {
      app.interacting_camera_controller = *camera_controller;
    } else {
      app.interacting_camera_controller.reset();
    }
  }
  ImGui::End();
  ImGui::PopStyleVar();
}
} // namespace pc::gui